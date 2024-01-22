counter=0
num_examples=40000
for exp_name in "gla"; do
    for ngram in 0 1 2 3; do
        for binary in "" "--binary"; do
            # if 1 and binary skip
            if [[ ${ngram} == 1 ]] && [[ ${binary} == "--binary" ]]; then
                echo "c1: skipping ${exp_name} ${ngram}gram ${binary}"
                continue
            fi
            # if 0 and not binary skip
            if [[ ${ngram} == 0 ]] && [[ ${binary} == "" ]]; then
                echo "c2: skipping ${exp_name} ${ngram}gram ${binary}"
                continue
            fi

            for hidden_key in "hidden_outputs"; do
                # if exp_name does not start with transformer, then skip attention_contexts
                if [[ ! ${exp_name} =~ transformer*  && ${exp_name} != "linear_transformer" ]] && [[ ${hidden_key} == "attention_contexts" ]]; then
                    echo "c3: skipping ${exp_name} ${hidden_key}"
                    continue
                fi
                exp_folder=interpretability_${num_examples}/${exp_name}/${ngram}gram/${binary//--/}/${hidden_key}/
                mkdir -p ${exp_folder}
                layer_no=${exp_name//transformer\//}
                # check if layer_no is a number
                if [[ ! ${layer_no} =~ ^[0-9]+$ ]]; then
                    layer_no=12
                fi
                for layer in `seq 0 ${layer_no}`; do
                    echo "${exp_name} ${ngram}gram ${binary} ${hidden_key} ${layer} counts"
                    export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=$((counter % 16)); python probe.py --layer ${layer} --use_wandb --ngram=${ngram} --exp="${exp_name}" --hidden_key=${hidden_key} ${binary} --num_examples=${num_examples} > ${exp_folder}/l${layer}.log 2>&1  &
                    counter=$((counter + 1))

                    if [[ ${counter} == 16 ]]; then
                        counter=10
                    fi

                    # if not binary and ngram != 0
                    if [[ ${binary} == "" ]] && [[ ${ngram} != 0 ]]; then
                        echo "${exp_name} ${ngram}gram ${binary} ${hidden_key} ${layer} ratio"
                        export PYTHONHASHSEED=0; export CUDA_VISIBLE_DEVICES=$((counter % 16)); python probe.py --layer ${layer} --use_wandb --ngram=${ngram} --exp="${exp_name}" --hidden_key=${hidden_key} ${binary}  --use_ratio --num_examples=${num_examples} > ${exp_folder}/l${layer}.log 2>&1  &
                        # echo "CUDA_VISIBLE_DEVICES=$((counter % 16)) python probe.py --layer ${layer} --use_wandb --ngram=${ngram} --exp=\"${exp_name}\" --hidden_key=${hidden_key} > ${exp_folder}/l${layer}.log 2>&1  &"
                        counter=$((counter + 1))
                    fi

                    if [[ ${counter} == 16 ]]; then
                        counter=10
                    fi

                    sleep 120
                done
                sleep 1000
            done
        done
    done
done
