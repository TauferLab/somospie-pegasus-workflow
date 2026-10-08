#!/usr/bin/env python3
"""Generate the somospie-pegasus-workflow Pegasus DAG from config.yml.

Reads data selection, model settings, and tiling from the config;
writes sites/transformations/replica catalogs + pegasus.properties +
workflow.yml. Submit with:
  pegasus-plan --submit -s condorpool -o local workflow.yml
(or -s local for submit-host testing).
"""

import argparse
import json
import logging
import os
import sys

import yaml

from Pegasus.api import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("somospie-generator")


# Per-transformation memory requests (matchmaking + OOM protection on
# shared HTCondor pools). Applied below via add_pegasus_profile
MEMORY = {
    "acquire_terrain": "2GB", "acquire_roi": "1GB",
    "acquire_soil_moisture": "1GB", "process_soil_moisture": "2GB", 
    "process_terrain": "2GB", "generate_train": "4GB", 
    "generate_eval": "4GB", "train_model": "4GB",
    "evaluate_model": "4GB", "merge_predictions": "2GB",
    "visualize_map": "2GB",
}

class SOMOSPIEWorkflow:
    def __init__(self, config_path, state=None, container_sif=None,
                 output=None, exec_site="condorpool", submit=False):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        if state:
            self.config["region"]["state"] = state
        self.wf_dir = os.path.dirname(os.path.abspath(config_path))
        self.config_path = os.path.abspath(config_path)
        self.container_sif = container_sif
        self.dagfile = output or os.path.join(
            self.wf_dir, self.config["workflow"]["output"])
        self.exec_site = exec_site
        self.submit = submit
        self.wf_name = self.config["workflow"]["name"]
        self.shared_scratch_dir = os.path.join(self.wf_dir, "scratch")
        self.local_storage_dir = os.path.join(self.wf_dir, "output")
        os.makedirs(self.shared_scratch_dir, exist_ok=True)
        os.makedirs(self.local_storage_dir, exist_ok=True)

        self.validate_config()
        self.derive()

     # ---- config ------------------------------------------------------
    def validate_config(self):
        c = self.config
        region = c.get("region", {})
        if not region.get("state") or not region.get("resolution"):
            raise ValueError("region.state and region.resolution required")
        roi_url = c.get("roi_url", "")
        if not (isinstance(roi_url, str) and roi_url.startswith("http")):
            raise ValueError("roi_url must be an http(s) URL")
        predictors = (c.get("data_sources", {}).get("terrain", {})
                      .get("predictors", {}))
        if not predictors:
            raise ValueError("at least one terrain predictor required")
        for name, filename in predictors.items():
            if not str(filename).endswith(".tif"):
                raise ValueError(f"predictor {name!r} must be a .tif file")
        sm = c.get("data_sources", {}).get("soil_moisture", {})
        months = sm.get("months", [])
        if not months or any(not 1 <= int(m) <= 12 for m in months):
            raise ValueError("soil_moisture.months must hold 1..12")
        train_month = c.get("training", {}).get("train_month")
        if int(train_month) not in [int(m) for m in months]:
            raise ValueError(f"training.train_month {train_month} not in "
                             f"soil_moisture.months {months}")
        n_tiles = c.get("inference", {}).get("n_tiles", 0)
        if int(n_tiles) < 1:
            raise ValueError("inference.n_tiles must be >= 1")
        tiles = self.eval_tiles()
        for x, y in tiles:
            if not (0 <= x < int(n_tiles) and 0 <= y < int(n_tiles)):
                raise ValueError(
                    f"eval tile ({x}, {y}) outside 0..{int(n_tiles) - 1}")
        viz = c.get("visualization", {})
        if (viz.get("vmin") is not None and viz.get("vmax") is not None
                and not viz["vmax"] > viz["vmin"]):
            raise ValueError("visualization.vmax must exceed vmin")   


    def eval_tiles(self):
        """Configured tiles as [(x, y)]; the string 'all' expands x-major."""
        inf = self.config["inference"]
        n = int(inf["n_tiles"])
        spec = inf["eval_tiles"]
        if isinstance(spec, str) and spec == "all":
            return [(x, y) for x in range(n) for y in range(n)]
        return [(int(x), int(y)) for x, y in spec]


    def derive(self):
        """Values computed from config (labels, names, ordered lists)."""
        c = self.config
        self.state = c["region"]["state"]
        self.resolution = c["region"]["resolution"]
        self.predictors = list(c["data_sources"]["terrain"]
                               ["predictors"].items())  # [(band, file)]
        year = c["data_sources"]["soil_moisture"]["year"]
        self.months = [int(m) for m in
                       c["data_sources"]["soil_moisture"]["months"]]
        tm = int(c["training"]["train_month"])
        self.train_label = f"{year}_{tm:02d}"
        self.tiles = self.eval_tiles()
        self.map_name = (f"predictions_{self.state}_"
                         f"{self.train_label}_map.tif")
        self.fig_name = (f"predictions_{self.state}_"
                         f"{self.train_label}_map.png")
        self.title = (f"Predicted soil moisture — {self.state} "
                      f"{year}-{tm:02d}")
        # Dataverse fallback slice -> flat {band: url}, warn on gaps.
        fb = (c["data_sources"]["terrain"].get("fallback") or {})
        self.fallback_map = {}
        if fb.get("enabled"):
            try:
                with open(os.path.join(self.wf_dir, fb.get("file", ""))) as f:
                    db = json.load(f)
                slice_ = db[self.resolution][self.state]
            except (OSError, ValueError, KeyError) as e:
                logger.warning(f"fallback file unusable: {e}")
                slice_ = {}
            for band, filename in self.predictors:
                url = slice_.get(filename, "")
                if not url or url == "REPLACE_ME":
                    logger.warning(
                        f"no Dataverse fallback for {filename}")
                else:
                    self.fallback_map[band] = url
        logger.info(f"state={self.state} train={self.train_label} "
                    f"tiles={len(self.tiles)} predictors={len(self.predictors)} "
                    f"fallback_urls={len(self.fallback_map)}")


    # ---- catalogs ----------------------------------------------------
    def create_pegasus_properties(self):
        self.props = Properties()
        self.props["pegasus.transfer.threads"] = "16"


    def create_sites_catalog(self):
        self.sc = SiteCatalog()
        local = Site("local").add_directories(
            Directory(Directory.SHARED_SCRATCH,
                      self.shared_scratch_dir).add_file_servers(
                          FileServer("file://" + self.shared_scratch_dir, Operation.ALL)
            ),
            Directory(Directory.LOCAL_STORAGE,
                      self.local_storage_dir).add_file_servers(
                          FileServer("file://" + self.local_storage_dir, Operation.ALL)
            ),
        )

        exec_site = (Site(self.exec_site)
                     .add_condor_profile(universe="vanilla")
                     .add_pegasus_profile(style="condor")
        )

        self.sc.add_sites(local, exec_site)


    def create_replica_catalog(self):
        """External inputs, recorded for provenance"""
        self.rc = ReplicaCatalog()
        for rel in ("config.yml", "dataverse.json"):
            path = os.path.join(self.wf_dir, rel)
            if os.path.isfile(path):
                self.rc.add_replica("local", File(rel), path)


    def create_transformation_catalog(self):
        self.tc = TransformationCatalog()
        container = None
        if self.container_sif:
            sif_path = (self.container_sif
                        if os.path.isabs(self.container_sif)
                        else os.path.join(self.wf_dir, self.container_sif))
            if not os.path.exists(sif_path):
                logger.warning(
                    f"container not found at {sif_path} — build it with: "
                    f"apptainer build {sif_path} Apptainer/SOMOSPIE_Container.def")
            container = Container(
                "somospie_container", Container.SINGULARITY,
                image="file://" + sif_path, image_site="local")
            self.tc.add_containers(container)
        scripts = {
            "acquire_terrain": "bin/acquire_terrain.py",
            "acquire_roi": "bin/acquire_roi.py",
            "acquire_soil_moisture": "bin/acquire_soil_moisture.py",
            "process_soil_moisture": "bin/process_soil_moisture.py",
            "process_terrain": "bin/process_terrain.py",
            "generate_train": "bin/generate_train.py",
            "generate_eval": "bin/generate_eval.py",
            "train_model": "bin/train_model.py",
            "evaluate_model": "bin/evaluate_model.py",
            "merge_predictions": "bin/merge_predictions.py",
            "visualize_map": "bin/visualize_map.py",
        }
        for name, rel in scripts.items():
            kwargs = {}
            if container is not None:
                kwargs["container"] = container
            tr = Transformation(
                name, site=self.exec_site,
                pfn=os.path.join(self.wf_dir, rel),
                is_stageable=True, **kwargs
            ).add_pegasus_profile(memory=MEMORY[name])
            self.tc.add_transformations(tr)


    # ---- jobs --------------------------------------------------------
    def _job(self, tr, job_id, args, inputs, outputs, stage_out_files=()):
        """args: argv list; inputs/outputs: File lists. Outputs in
        stage_out_files land in output/; the rest stay in scratch."""
        job = Job(tr, _id=job_id, node_label=job_id)
        job.add_args(*args)
        for f in inputs:
            job.add_inputs(f)
        for f in outputs:
            job.add_outputs(f, stage_out=(f.lfn in stage_out_files),
                            register_replica=False)
        job.add_pegasus_profile(label=job_id)
        self.wf.add_jobs(job)
        return job


    def create_workflow(self):
        c = self.config
        terrain = c["data_sources"]["terrain"]
        sm = c["data_sources"]["soil_moisture"]
        self.wf = Workflow(self.wf_name)

        # Acquire terrain (predictors only; ROI comes from acquire_roi).
        terrain_files = [File(fn) for _, fn in self.predictors]
        terrain_manifest = File("terrain_manifest.json")
        acquire_terrain = self._job(
            "acquire_terrain", "acquire_terrain",
            ["--service-url", terrain["service_url"],
             "--state", self.state, 
             "--resolution", self.resolution]
            + [a for band, fn in self.predictors
               for a in ("--want", f"{band}={fn}")]
            + ["--output-dir", ".", 
               "--manifest", terrain_manifest.lfn,
               "--fallback-map", json.dumps(self.fallback_map)],
            [], terrain_files + [terrain_manifest],
            stage_out_files={terrain_manifest.lfn})

        # Acquire ROI (GADM state boundary for the cutline)
        roi_zip = File(f"state_{self.state}.zip")
        roi_manifest = File("roi_manifest.json")
        acquire_roi = self._job(
            "acquire_roi", "acquire_roi",
            ["--gadm-url", c["roi_url"], 
             "--state", self.state,
             "--output-dir", ".", 
             "--output-zip", roi_zip.lfn,
             "--manifest", roi_manifest.lfn],
            [], [roi_zip, roi_manifest],
            stage_out_files={roi_manifest.lfn})

        # Acquire + process soil moisture, one pair per month.
        sm_tifs = {}
        proc_jobs = {}
        for month in self.months:
            label = f"{sm['year']}_{month:02d}"
            days_zip = File(f"sm_{label}_days.zip")
            days_manifest = File(f"sm_{label}_manifest.json")
            monthly = File(f"sm_{label}.tif")
            sm_tifs[month] = monthly
            acq = self._job(
                "acquire_soil_moisture", f"acquire_sm_{label}",
                ["--ftp-base", sm["base_url"],
                 "--version", str(sm["version"]),
                 "--year", str(sm["year"]),
                 "--month", str(month),
                 "--output-dir", ".",
                 "--days-zip", days_zip.lfn,
                 "--manifest", days_manifest.lfn],
                [], [days_zip, days_manifest],
                stage_out_files={days_manifest.lfn})
            
            proc = self._job(
                "process_soil_moisture", f"process_sm_{label}",
                ["--days-zip", days_zip.lfn,
                 "--month-manifest", days_manifest.lfn,
                 "--output", monthly.lfn],
                [days_zip, days_manifest], [monthly], set())
            proc_jobs[month] = proc
            self.wf.add_dependency(acq, children=[proc])

        # Process terrain (reprojected predictors + cutline).
        proc_files = [File("proc_" + fn) for _, fn in self.predictors]
        cutline = File(f"proc_state_{self.state}_cutline.zip")
        proc_manifest = File("terrain_proc_manifest.json")
        process_terrain = self._job(
            "process_terrain", "process_terrain",
            ["--input-dir", ".", "--param-names"] + [b for b, _ in self.predictors] 
            + ["--param-files"] + [f.lfn for f in terrain_files] 
            + ["--roi-zip", roi_zip.lfn, 
               "--out-prefix", "proc_",
               "--output-dir", ".", 
               "--manifest", proc_manifest.lfn],
            terrain_files + [roi_zip],
            proc_files + [cutline, proc_manifest],
            stage_out_files={proc_manifest.lfn})
        self.wf.add_dependency(acquire_terrain, children=[process_terrain])
        self.wf.add_dependency(acquire_roi, children=[process_terrain])

        # Training raster for the configured month.
        train_tif = File(f"train_{self.train_label}.tif")
        tm_month = int(c["training"]["train_month"])
        generate_train = self._job(
            "generate_train", "generate_train",
            ["-i", sm_tifs[tm_month].lfn,
             "-f"] + [f.lfn for f in proc_files] + [
             "-p"] + [b for b, _ in self.predictors] + [
             "-s", cutline.lfn, "-o", train_tif.lfn],
            [sm_tifs[tm_month]] + proc_files + [cutline],
            [train_tif], set())
        self.wf.add_dependency(process_terrain, children=[generate_train])
        self.wf.add_dependency(proc_jobs[tm_month], children=[generate_train])

        # Eval tiles (one job emits all configured tiles)
        eval_files = [File(f"eval_{i:04d}.tif")
                      for i in range(len(self.tiles))]
        tile_args = [a for (x, y) in self.tiles
                     for a in ("--tile", str(x), str(y))]
        generate_eval = self._job(
            "generate_eval", "generate_eval",
            ["-i"] + [f.lfn for f in proc_files] + [
             "-p"] + [b for b, _ in self.predictors] + [
             "-n", str(c["inference"]["n_tiles"])] + tile_args + [
             "-s", cutline.lfn, "--output-dir", "."],
            proc_files + [cutline], eval_files, set())
        self.wf.add_dependency(process_terrain, children=[generate_eval])

        # Train (RF only)
        tr = c["training"]
        model_file = File("model.pkl")
        scaler_file = File("scaler.pkl")
        metrics_file = File("metrics.json")
        train_model = self._job(
            "train_model", "train_model",
            ["-i", train_tif.lfn, 
             "-o", model_file.lfn,
             "-s", scaler_file.lfn, 
             "-m", metrics_file.lfn,
             "--max-trees", str(tr["max_trees"]),
             "--n-iter", str(tr["n_iter"]), 
             "--cv", str(tr["cv"]),
             "--seed", str(tr["seed"])],
            [train_tif],
            [model_file, scaler_file, metrics_file],
            stage_out_files={model_file.lfn, scaler_file.lfn,
                             metrics_file.lfn})
        self.wf.add_dependency(generate_train, children=[train_model])

        # Inference fan-out (one job per tile; empty tiles yield nodata tiles)
        pred_files = []
        eval_jobs = []
        for i in range(len(self.tiles)):
            pred = File(f"predictions_{i:04d}.tif")
            pred_files.append(pred)
            job = self._job(
                "evaluate_model", f"evaluate_{i:04d}",
                ["-i", eval_files[i].lfn,
                 "-o", pred.lfn,
                 "-s", scaler_file.lfn,
                 "-m", model_file.lfn],
                [eval_files[i], scaler_file, model_file],
                [pred],
                {pred.lfn})
            self.wf.add_dependency(generate_eval, children=[job])
            self.wf.add_dependency(train_model, children=[job])
            eval_jobs.append(job)

        # Mosaic predicted tiles into the state map
        map_file = File(self.map_name)
        merge = self._job(
            "merge_predictions", "merge_predictions",
            ["-i"] + [f.lfn for f in pred_files] + ["-o", map_file.lfn],
            pred_files, [map_file], {map_file.lfn})
        for job in eval_jobs:
            self.wf.add_dependency(job, children=[merge])

        # Render the map figure
        viz = c.get("visualization", {})
        fig_file = File(self.fig_name)
        viz_args = ["--input", map_file.lfn, 
                    "--output", fig_file.lfn,
                    "--cmap", str(viz.get("cmap", "RdBu")),
                    "--dpi", str(viz.get("dpi", 150)),
                    "--title", self.title]
        if viz.get("vmin") is not None:
            viz_args += ["--vmin", str(viz["vmin"])]
        if viz.get("vmax") is not None:
            viz_args += ["--vmax", str(viz["vmax"])]
        visualize = self._job(
            "visualize_map", "visualize_map", viz_args,
            [map_file], [fig_file], {fig_file.lfn})
        self.wf.add_dependency(merge, children=[visualize])


    def write(self):
        if self.sc is not None:
            self.sc.write()
        self.props.write()
        self.rc.write()
        self.tc.write()
        self.wf.write(file=self.dagfile)

    def build(self):
        self.create_pegasus_properties()
        self.create_sites_catalog()
        self.create_replica_catalog()
        self.create_transformation_catalog()
        self.create_workflow()
        self.write()
        logger.info(f"wrote {self.dagfile} (+ sites/transforms/replicas.yml, "
                    f"pegasus.properties)")
        if self.submit:
            self.wf.plan(sites=[self.exec_site], output_sites=["local"],
                         submit=True)
            self.wf.wait()
            self.wf.statistics()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate the SOMOSPIE Pegasus DAG from config.yml.")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--state", default=None,
                        help="override region.state (e.g. AL)")
    parser.add_argument("--container-sif", default=None,
                        help="run jobs in this .sif (omit = bare interpreter)")
    parser.add_argument("--output", default=None,
                        help="DAG path (default from config)")
    parser.add_argument("-e", "--exec-site", default="condorpool",
                        help="execution site: condorpool | local")
    parser.add_argument("--submit", action="store_true",
                        help="plan, submit, wait, and print statistics")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.config):
        print(f"error: config not found: {args.config}", file=sys.stderr)
        return 2
    try:
        wf = SOMOSPIEWorkflow(args.config, state=args.state,
                              container_sif=args.container_sif,
                              output=args.output,
                              exec_site=args.exec_site,
                              submit=args.submit)
        wf.build()
    except (ValueError, KeyError) as e:
        print(f"error: invalid config: {e}", file=sys.stderr)
        return 1
    print(f"next: pegasus-plan --submit -s {args.exec_site} -o local "
          f"{wf.dagfile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


    