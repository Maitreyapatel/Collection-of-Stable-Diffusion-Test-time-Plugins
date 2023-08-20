CUDA_VISIBLE_DEVICES=0 python main.py \
    --exp_name="lg" \
    --lg.prompt="a cat and a dog" \
    --lg.phrases="cat;dog" --lg.bounding_box="[[[0.1, 0.2, 0.5, 0.8]],[[0.75, 0.6, 0.95, 0.8]]]" --lg.attention_aggregation_method="all_attention" \
    --lg.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=1 python main.py \
    --exp_name="lg" \
    --lg.prompt="a apple and a frog" \
    --lg.phrases="apple;frog" --lg.bounding_box="[[[0.1, 0.2, 0.5, 0.8]],[[0.75, 0.6, 0.95, 0.8]]]" --lg.attention_aggregation_method="all_attention" \
    --lg.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=2 python main.py \
    --exp_name="lg" \
    --lg.prompt="a car and a sheep" \
    --lg.phrases="car;sheep" --lg.bounding_box="[[[0.1, 0.2, 0.5, 0.8]],[[0.75, 0.6, 0.95, 0.8]]]" --lg.attention_aggregation_method="all_attention" \
    --lg.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=3 python main.py \
    --exp_name="lg" \
    --lg.prompt="a bench and a boat" \
    --lg.phrases="bench;boat" --lg.bounding_box="[[[0.1, 0.2, 0.5, 0.8]],[[0.75, 0.6, 0.95, 0.8]]]" --lg.attention_aggregation_method="all_attention" \
    --lg.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=4 python main.py \
    --exp_name=lg \
    --lg.prompt="a bird and a clock" \
    --lg.phrases="bird;clock" --lg.bounding_box="[[[0.1, 0.2, 0.5, 0.8]],[[0.75, 0.6, 0.95, 0.8]]]" --lg.attention_aggregation_method="all_attention" \
    --lg.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=5 python main.py \
    --exp_name="lg" \
    --lg.prompt="a cup and a vase" \
    --lg.phrases="cup;vase" --lg.bounding_box="[[[0.1, 0.2, 0.5, 0.8]],[[0.75, 0.6, 0.95, 0.8]]]" --lg.attention_aggregation_method="all_attention" \
    --lg.seeds=[42,1989,293850] &

CUDA_VISIBLE_DEVICES=6 python main.py \
    --exp_name="lg" \
    --lg.prompt="a vase and a book" \
    --lg.phrases="vase;book" --lg.bounding_box="[[[0.1, 0.2, 0.5, 0.8]],[[0.75, 0.6, 0.95, 0.8]]]" --lg.attention_aggregation_method="all_attention" \
    --lg.seeds=[42,1989,293850] &

wait

CUDA_VISIBLE_DEVICES=6 python main.py \
    --exp_name="lg" \
    --lg.prompt="a red car and a white sheep" \
    --lg.phrases="red car;white sheep" --lg.bounding_box="[[[0.1, 0.2, 0.5, 0.8]],[[0.75, 0.6, 0.95, 0.8]]]" --lg.attention_aggregation_method="all_attention" \
    --lg.seeds=[42,1989,293850] &

wait