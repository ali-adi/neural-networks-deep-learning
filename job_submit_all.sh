#!/bin/bash

# Helper to wait for user jobs to drop below SLURM limit
wait_for_slot() {
    while true; do
        count=$(squeue -u $USER | grep -c 'UGGPU-TC1')
        if [ "$count" -lt 2 ]; then
            break
        fi
        echo "Waiting for slot... (currently $count jobs)"
        sleep 60
    done
}

submit_job() {
    script=$1
    name=$2
    echo "Submitting $script"
    jid=$(sbatch $script | awk '{print $4}')
    echo "$name submitted with JobID: $jid"
    echo $jid
}

# Start submission
jid1=$(submit_job job1_train_emodb.sh "Job 1")
wait_for_slot

jid2=$(submit_job job2_train_ravdess.sh "Job 2")
wait_for_slot

jid3=$(submit_job job3_da_emodb.sh "Job 3")
wait_for_slot

jid4=$(submit_job job4_da_ravdess.sh "Job 4")
wait_for_slot

# Submit job5 and job6 with dependencies on jobs 1–4
echo "Submitting Job 5 (Same-corpus Testing)"
jid5=$(sbatch --dependency=afterok:$jid1:$jid2:$jid3:$jid4 job5_test_same.sh | awk '{print $4}')
echo "Job 5 submitted with JobID: $jid5"

echo "Submitting Job 6 (Cross-corpus Testing)"
sbatch --dependency=afterok:$jid1:$jid2:$jid3:$jid4 job6_test_cross.sh
