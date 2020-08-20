np=$1
config_file=$2
epit=$3
maxeval=$4
initfile=$5
Job=$6
nxfar=1
nyfar=1
nzfar=1
dxfar=0.01
dyfar=0.01
dzfar=0.01

mpirun -np $np strehlopt_exec -options_file $config_file \
       -filter_radius 5 \
       -filter_sigma 10 \
       -filter_beta 60 \
       -cutoff 0.01 \
       -Job $Job \
       -epi_t $epit \
       -algouter 24 \
       -alginner 24 \
       -algmaxeval $maxeval \
       -autoinit 0.5 \
       -initial_filename $initfile \
       -printeff 0 \
       -nx_far $nxfar \
       -ny_far $nyfar \
       -nz_far $nzfar \
       -dx_far $dxfar \
       -dy_far $dyfar \
       -dz_far $dzfar \
       #-fwhm 25 \
       #-s0,s1,ds 0,1,0.001 \       
       

