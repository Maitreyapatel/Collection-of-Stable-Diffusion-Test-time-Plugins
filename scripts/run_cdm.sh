CUDA_VISIBLE_DEVICES=0 python main.py \
    --exp_name="cdm" \
    --cdm.prompt="a cat and a dog" \
    --cdm.prompt_a="a photo of a dog" \
    --cdm.prompt_b="a photo of a cat" \
    --cdm.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=1 python main.py \
    --exp_name="cdm" \
    --cdm.prompt="a apple and a frog" \
    --cdm.prompt_a="a photo of a apple" \
    --cdm.prompt_b="a photo of a frog" \
    --cdm.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=2 python main.py \
    --exp_name="cdm" \
    --cdm.prompt="a car and a sheep" \
    --cdm.prompt_a="a photo of a car" \
    --cdm.prompt_b="a photo of a sheep" \
    --cdm.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=3 python main.py \
    --exp_name="cdm" \
    --cdm.prompt="a bench and a boat" \
    --cdm.prompt_a="a photo of a bench" \
    --cdm.prompt_b="a photo of a boat" \
    --cdm.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=4 python main.py \
    --exp_name=cdm \
    --cdm.prompt="a bird and a clock" \
    --cdm.prompt_a="a photo of a bird" \
    --cdm.prompt_b="a photo of a clock" \
    --cdm.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=5 python main.py \
    --exp_name="cdm" \
    --cdm.prompt="a cup and a vase" \
    --cdm.prompt_a="a photo of a cup" \
    --cdm.prompt_b="a photo of a vase" \
    --cdm.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=6 python main.py \
    --exp_name="cdm" \
    --cdm.prompt="a vase and a book" \
    --cdm.prompt_a="a photo of a vase" \
    --cdm.prompt_b="a photo of a book" \
    --cdm.seeds=[42,1989,293850] &

wait

CUDA_VISIBLE_DEVICES=6 python main.py \
    --exp_name="cdm" \
    --cdm.prompt="a red car and a white sheep" \
    --cdm.prompt_a="a red car" \
    --cdm.prompt_b="a white sheep" \
    --cdm.seeds=[42,1989,293850] &

wait
