counter=12
num_examples=2500
for exp_name in "transformer/1"; do
    for ngram in 1 2 3; do
        for binary in "" "--binary"; do
            # if 1 and binary skip
            if [[ ${ngram} == 1 ]] && [[ ${binary} == "--binary" ]]; then
                continue
            fi
            # if 0 and not binary skip
            if [[ ${ngram} == 0 ]] && [[ ${binary} == "" ]]; then
                continue
            fi

            for hidden_key in "attention_contexts"; do
                # if exp_name does not start with transformer, then skip attention_contexts
                # if [[ ! ${exp_name} =~ "transformer*" ]] && [[ ${hidden_key} == "attention_contexts" ]]; then
                #     echo "skipping attention_contexts for ${exp_name} ${hidden_key}"
                #     continue
                # fi
                exp_folder=interpretability_${num_examples}/${exp_name}/${ngram}gram/${binary//--/}/${hidden_key}/
                mkdir -p ${exp_folder}
                layer_no=${exp_name//transformer\//}
                # check if layer_no is a number
                if [[ ! ${layer_no} =~ ^[0-9]+$ ]]; then
                    layer_no=8
                    echo "layer_no is not a number, setting it to 4 for ${exp_name}"
                fi
                for layer in `seq 0 ${layer_no}`; do
                    export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=$((counter % 16)); python probe.py --layer ${layer} --use_wandb --ngram=${ngram} --exp="${exp_name}" --hidden_key=${hidden_key} ${binary} --num_examples=${num_examples} > ${exp_folder}/l${layer}.log 2>&1  &
                    # echo "CUDA_VISIBLE_DEVICES=$((counter % 16)) python probe.py --layer ${layer} --use_wandb --ngram=${ngram} --exp=\"${exp_name}\" --hidden_key=${hidden_key} > ${exp_folder}/l${layer}.log 2>&1  &"
                    counter=$((counter + 1))

                    # if not binary and ngram != 0
                    if [[ ${binary} == "" ]] && [[ ${ngram} != 0 ]]; then
                        export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=$((counter % 16)); python probe.py --layer ${layer} --use_wandb --ngram=${ngram} --exp="${exp_name}" --hidden_key=${hidden_key} ${binary}  --use_ratio --num_examples=${num_examples}  > ${exp_folder}/l${layer}.log 2>&1  &
                        # echo "CUDA_VISIBLE_DEVICES=$((counter % 16)) python probe.py --layer ${layer} --use_wandb --ngram=${ngram} --exp=\"${exp_name}\" --hidden_key=${hidden_key} > ${exp_folder}/l${layer}.log 2>&1  &"
                        counter=$((counter + 1))
                    fi

                    # if counter < 12, then set it to 12
                    if [ $((counter)) -lt 12 ]; then
                        counter=12
                    fi



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

            done
            sleep 20
        done
    done
    # sleep for 15 minutes
    sleep 600
done
