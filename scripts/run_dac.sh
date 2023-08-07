CUDA_VISIBLE_DEVICES=0 python main.py \
    --exp_name="dac" \
    --dac.prompt="a cat and a dog" \
    --dac.prompt_a="a photo of a dog" \
    --dac.token_a="dog" \
    --dac.prompt_b="a photo of a cat" \
    --dac.token_b="cat" \
    --dac.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=1 python main.py \
    --exp_name="dac" \
    --dac.prompt="a apple and a frog" \
    --dac.prompt_a="a photo of a apple" \
    --dac.token_a="apple" \
    --dac.prompt_b="a photo of a frog" \
    --dac.token_b="frog" \
    --dac.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=2 python main.py \
    --exp_name="dac" \
    --dac.prompt="a car and a sheep" \
    --dac.prompt_a="a photo of a car" \
    --dac.token_a="car" \
    --dac.prompt_b="a photo of a sheep" \
    --dac.token_b="sheep" \
    --dac.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=3 python main.py \
    --exp_name="dac" \
    --dac.prompt="a bench and a boat" \
    --dac.prompt_a="a photo of a bench" \
    --dac.token_a="bench" \
    --dac.prompt_b="a photo of a boat" \
    --dac.token_b="boat" \
    --dac.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=4 python main.py \
    --exp_name=dac \
    --dac.prompt="a bird and a clock" \
    --dac.prompt_a="a photo of a bird" \
    --dac.token_a="bird" \
    --dac.prompt_b="a photo of a clock" \
    --dac.token_b="clock" \
    --dac.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=5 python main.py \
    --exp_name="dac" \
    --dac.prompt="a cup and a vase" \
    --dac.prompt_a="a photo of a cup" \
    --dac.token_a="cup" \
    --dac.prompt_b="a photo of a vase" \
    --dac.token_b="vase" \
    --dac.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=6 python main.py \
    --exp_name="dac" \
    --dac.prompt="a vase and a book" \
    --dac.prompt_a="a photo of a vase" \
    --dac.token_a="vase" \
    --dac.prompt_b="a photo of a book" \
    --dac.token_b="book" \
    --dac.seeds=[42,1989,293850] &

wait