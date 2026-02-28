# python compute_percentiles.py --ground_truth ../data/experiment/data/datastore.interior.domain03.zarr --output percentiles_iconles.nc --margin 0.005
# python compute_percentiles.py --predictions Baseline=../evals/baseline_unroll30.zarr --output percentiles_baseline.nc --margin 0.005
# python compute_percentiles.py --predictions Sfconly=../evals/sfconly_unroll30.zarr --output percentiles_sfconly.nc --margin 0.005
# python compute_percentiles.py --predictions bare-minimum=../evals/bare-minimum_unroll30.zarr --output percentiles_bare_minimum.nc --margin 0.005
# python compute_percentiles.py --predictions sw=../evals/sw_unroll30.zarr --output percentiles_sw.nc --margin 0.005
python compute_percentiles.py --predictions qv=../evals/qv_unroll30.zarr --output percentiles_qv.nc2 --margin 0.005
# python compute_percentiles.py --predictions lw=../evals/lw_unroll30.zarr --output percentiles_lw.nc --margin 0.005
python compute_percentiles.py --predictions swlw=../evals/swlw_unroll30.zarr --output percentiles_swlw.nc2 --margin 0.005
# python compute_percentiles.py --predictions rr=../evals/rain_unroll30.zarr --output percentiles_rr.nc --margin 0.005

