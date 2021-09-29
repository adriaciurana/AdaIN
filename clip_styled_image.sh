# python clip_styled_image.py \
#     --weights_path "/media/totolia/datos_3/research/adain/results/model_with_diff_loss/26_09_2021__20_01_52/checkpoints/checkpoint_192_0.757894.pkl" \
#     --image_path "/media/totolia/datos_3/research/adain/test_images/content/hinton.jpg" \
#     --target_text "human with blue shirt" \
#     --alpha 1.0 \
#     --loss_type directional \
#     --source_text "human face"

python clip_styled_image.py \
    --weights_path "/media/totolia/datos_3/research/adain/results/model_with_diff_loss/26_09_2021__20_01_52/checkpoints/checkpoint_192_0.757894.pkl" \
    --image_path "/media/totolia/datos_3/research/adain/test_images/content/hinton.jpg" \
    --target_text "black human face" \
    --alpha 1.0 \
    --loss_type directional \
    --source_text "white human face"
