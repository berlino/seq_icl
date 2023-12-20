counter=0
for exp_name in "transformer/12"; do
    for ngram in 3; do
        for binary in ""; do
            for hidden_key in "hidden_outputs" "attention_contexts"; do
                exp_folder=interpretability/${exp_name}/${ngram}gram/${binary//--/}/${hidden_key}/
                mkdir -p ${exp_folder}
                for layer in `seq 0 12`; do
                    export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=$((counter % 16)); python probe.py --layer ${layer} --use_wandb --ngram=${ngram} --exp="${exp_name}" --hidden_key=${hidden_key} ${binary} > ${exp_folder}/l${layer}.log 2>&1  &
                    counter=$((counter + 1))
                    # if [ $((counter)) -eq 0 ]; then
                    #     counter=1
                    # fi
                    # if [ $((counter)) -eq 6 ]; then
                    #     counter=8
                    # fi
                    # if [ $((counter)) -eq 7 ]; then
                    #     counter=8
                    # fi
                    # if [ $((counter)) -eq 15 ]; then
                    #     counter=1
                    # fi

                done
                sleep 20
            done
        done
        # sleep for 15 minutes
        # sleep 1000
    done
done
