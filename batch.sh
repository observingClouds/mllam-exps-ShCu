#!/bin/bash

variables=("clct" "lhfl_s" "pres_sfc" "qv_2m" "rain_gsp_rate" "shfl_s" "t_2m" "t_seasfc" "tot_prec" "tqc_dia" "tqi_dia" "tqv_dia" "u_10m" "v_10m" "sob_t" "sod_t" "sou_t" "thb_t" "synsat_rttov_forward_model_2__abi_ir__goes_16__channel_7")                                                                                                   

for var in "${variables[@]}"; do
    uv run python src/preprocess.py --var=$var --output=/home/has/data/sources/icon/ICON312_v1.0.0.zarr
done

