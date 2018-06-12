import Augmentor
pipeline = Augmentor.Pipeline("/Users/kpentchev/artmimir/meisje_met_de_parel")
pipeline.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
pipeline.zoom(probability=0.8, min_factor=1.1, max_factor=1.5)
pipeline.random_brightness(probability=0.8, min_factor=0.1, max_factor=0.9)
pipeline.crop_random(probability=0.5, percentage_area=0.5)
pipeline.skew(0.8, 0.9)

pipeline.sample(3000)
pipeline.process()