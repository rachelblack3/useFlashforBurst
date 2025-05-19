#!/bin/bash

## remove all data files and output from the hpc folders
rm -rf /data/hpcflash/users/rablack75/data/L4/*
rm -rf //data/hpcflash/users/rablack75/data/LANL/*
rm -rf /data/hpcflash/users/rablack75/data/magnetometer/*
rm -rf /data/hpcflash/users/rablack75/data/Survey/*
rm -rf /data/hpcflash/users/rablack75/data/WNA-Survey/*
rm -rf /data/hpcflash/users/rablack75/data/Burst/*
rm -rf /data/hpcflash/users/rablack75/power_netCDFs/*
wait

startdate="20160101"
enddate="20171201"

currdate=$startdate

## Looping by year until enddate is reached
while [[ "$currdate" < "$enddate" ]]; do

    ## Get the year and the month from the current date
    year=$(date -d "$currdate" +%Y)
    month=$(date -d "$currdate" +%m)
    day=$(date -d "$currdate" +%d)

    ## Copy all of the day folders to the HPCDATA filesystem
    HPCDATA='/data/hpcflash/users/rablack75/data'

    SURVEYDATA='/data/spacecast/wave_database_v2/RBSP-A/L3/$year/*'

    ## copy over magnetometer and L4 data for the year 
    scp -r /data/spacecast/wave_database_v2/RBSP-A/L3/$year/* /data/hpcflash/users/rablack75/data/magnetometer &
    scp -r /data/spacecast/satellite/RBSP/emfisis/data/RBSP-A/L4/$year/* /data/hpcflash/users/rablack75/data/L4 &
    scp -r /data/emfisis_burst/wip/rablack75/rablack75/SLURM_outA/$year/* /data/hpcflash/users/rablack75/data/Burst &

    ## copy over magnetometer data for the year 
    mag_source=/data/spacecast/satellite/RBSP/emfisis/data/RBSP-A/LANL/MagEphem/$year
    mag_dest=/data/hpcflash/users/rablack75/data/LANL
    prefix="rbspa_def_MagEphem_TS04D_"
    file_ending=".h5"
    find "$mag_source" -type f -name "${prefix}*${file_ending}" -exec rsync -av {} "$mag_dest" \;


    ## copy over survey for the year
    survey_source=/data/spacecast/wave_database_v2/RBSP-A/L2/$year 
    survey_dest=/data/hpcflash/users/rablack75/data/Survey
    prefix="rbsp-a_WFR-spectral-matrix-diagonal_emfisis-L2_"
    file_ending=".cdf"
    find "$survey_source" -type f -name "${prefix}*${file_ending}" -exec rsync -av {} "$survey_dest" \;


    # Wait for all background processes to complete
    wait
    
    # All files have been copied, proceed with the next commands
    echo "All magnetometer and L4 files have been copied."
    Survey_months=(/data/hpcflash/users/rablack75/data/L4/*) 
    
    ## Looping by month until enddate is reached
    curryear=$(date -d "$currdate" +%Y)
    nextyear=$(date -d "$currdate + 1year" +%Y)

    while [[ "$curryear" -le "$nextyear" ]]; do
        curryear=$(date -d "$currdate" +%Y)
        month=$(date -d "$currdate" +%m)

        ## Name variable containing all burst days
        Survey_days=(/data/hpcflash/users/rablack75/data/L4/$month/*) 
        ## Get number of days in month from length of Survey_days                                     
        numdays=${#Survey_days[@]} 
        echo $numdays

        ## activate python envrioment
        source /data/hpcdata/users/rablack75/burstenv/bin/activate
        ## deactivate python enviroment
        deactivate

        # take off 1 day from number of days as the array indicies in slurm script goes from 0 to numdays-1 
        numdays=`expr $((numdays-1))`

        ## Run actual sbatch processing code for each day of the month
        ## the --export options allows you to list bash varaiables that you wish to pass to the slurm script
        ## the --array option sets the slurm array, where each index is a day
        sbatch --array=0-$numdays --export=currdate=$currdate processing_batch_file.sh

        ## Increment the date --> next month
        currdate=$(date -d "$currdate + 1month" +%Y%m%d)
        echo "${year} ${month} is done"
        curryear=$(date -d "$currdate" +%Y)
        echo "${curryear} and ${nextyear}"

    done   

    ## create folder for daily output
    mkdir -p /data/emfisis_burst/wip/rablack75/rablack75/CountBurst/CSVs_flash/$year/
    
    ## transfer all new files to /data/emfisisburst
    scp -r /data/hpcflash/users/rablack75/power_netCDFs/* /data/emfisis_burst/wip/rablack75/rablack75/CountBurst/CSVs_flash/$year

    ## remove all data files and output from the hpc folders
    rm -rf /data/hpcflash/users/rablack75/data/L4/*
    rm -rf /data/hpcflash/users/rablack75/data/LANL/*
    rm -rf /data/hpcflash/users/rablack75/data/magnetometer/*
    rm -rf /data/hpcflash/users/rablack75/data/Survey/*
    rm -rf /data/hpcflash/users/rablack75/data/WNA-Survey/*
    rm -rf /data/hpcflash/users/rablack75/data/Burst/*
    rm -rf /data/hpcflash/users/rablack75/power_netCDFs/*
   
    
    ## Increment the date --> next year
    echo "${year} is done"
    
done