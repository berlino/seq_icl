counter=0
for exp_name in "rwkv"; do
    for ngram in 0 2 3; do
        for binary in "--binary"; do
            for hidden_key in "hidden_outputs" "attention_contexts"; do
                # if exp_name does not start with transformer, then skip attention_contexts
                if [[ ! ${exp_name} =~ "transformer*" ]] && [[ ${hidden_key} == "attention_contexts" ]]; then
                    continue
                    echo "skipping attention_contexts for ${exp_name} ${hidden_key}"
                fi
                exp_folder=interpretability_2500/${exp_name}/${ngram}gram/${binary//--/}/${hidden_key}/
                mkdir -p ${exp_folder}
                layer_no=${exp_name//transformer\//}
                # check if layer_no is a number
                if [[ ! ${layer_no} =~ ^[0-9]+$ ]]; then
                    layer_no=4
                    echo "layer_no is not a number, setting it to 4 for ${exp_name}"
                fi
                for layer in `seq 0 ${layer_no}`; do
                    export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=$((counter % 16)); python probe.py --layer ${layer} --use_wandb --ngram=${ngram} --exp="${exp_name}" --hidden_key=${hidden_key} ${binary} --use_ratio > ${exp_folder}/l${layer}.log 2>&1  &
                    # echo "CUDA_VISIBLE_DEVICES=$((counter % 16)) python probe.py --layer ${layer} --use_wandb --ngram=${ngram} --exp=\"${exp_name}\" --hidden_key=${hidden_key} > ${exp_folder}/l${layer}.log 2>&1  &"
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
    sleep 600
done
