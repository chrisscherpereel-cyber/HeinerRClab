"""Research runner: studies run from the command line, independent of Streamlit, with auditable, resumable records.

    python -m heiner_abm.research list                       studies and their evidence category
    python -m heiner_abm.research config <study>             the default configuration (JSON)
    python -m heiner_abm.research run <study> [--config f]   run (or resume) a study; prints the run directory
    python -m heiner_abm.research verify <run>               check every checksum of a stored run
    python -m heiner_abm.research status                     claims and whether their evidence is current
    python -m heiner_abm.research manuscript                 regenerate tables and figures from recorded runs

Modules:
    provenance   environment, git commit, dependency versions, source fingerprints, checksums, canonical JSON
    store        run directories, write-once stages, manifests, resume, export
    traces       decision-level traces of tournament agents (observation, prediction, action, feedback timing)
    studies      the studies the runner can execute, each as a sequence of stages
    claims       the claim-to-evidence registry
    manuscript   tables and figures regenerated from recorded runs
"""
