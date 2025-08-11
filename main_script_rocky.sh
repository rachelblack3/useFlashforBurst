#!/bin/bash
#mkdir  /data/emfisis_burst/wip/rablack75/rablack75/CountBurst/CSVs_flashBV2
## remove all data files and output from the hpc folders
rm -rf /data/hpcflash/users/rablack75/power_netCDFs/*

wait

startdate="20130101"
enddate="20140101"

currdate=$startdate

## Looping by year until enddate is reached
while [[ "$currdate" < "$enddate" ]]; do

    # All files have been copied, proceed with the next commands
    echo "Lets begin."
    
    ## Looping by month until enddate is reached
    curryear=$(date -d "$currdate" +%Y)
    nextyear=$(date -d "$currdate + 1year" +%Y)

    while [[ "$curryear" -lt "$nextyear" ]]; do
        curryear=$(date -d "$currdate" +%Y)
        month=$(date -d "$currdate" +%m)
        year=$(date -d "$currdate" +%Y)
        ## Name variable containing all burst days
        # if not using rocky`: Survey_days=(/data/hpcflash/users/rablack75/data/L4/$month/*) 
        Survey_days=(/data/spacecast/satellite/RBSP/emfisis/data/RBSP-B/L4/$year/$month/*) 
        ## Get number of days in month from length of Survey_days                                     
        numdays=${#Survey_days[@]} 
        echo $numdays

        # take off 1 day from number of days as the array indicies in slurm script goes from 0 to numdays-1 
        numdays=`expr $((numdays-1))`

        ## Run actual sbatch processing code for each day of the month
        ## the --export options allows you to list bash varaiables that you wish to pass to the slurm script
        ## the --array option sets the slurm array, where each index is a day
        sbatch --array=0-$numdays --export=currdate=$currdate processing_batch_file.sh

        ## Increment the date --> next month
        currdate=$(date -d "$currdate + 1month" +%Y%m%d)
        curryear=$(date -d "$currdate" +%Y)
        echo "${curryear} ${month} is done"
        
        
        echo "${curryear} and ${nextyear}"

    done   
  
   
    
    ## transfer all new files to /data/emfisisburst
    scp -r /data/hpcflash/users/rablack75/power_netCDFs/* /data/emfisis_burst/wip/rablack75/rablack75/CountBurst/CSVs_flashBV2/

    ## remove all data files and output from the hpc folders
    rm -rf /data/hpcflash/users/rablack75/power_netCDFs/*
   
    ## Increment the date --> next year
    echo "${year} is done"
    
done