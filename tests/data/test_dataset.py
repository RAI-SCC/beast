from conftest import dist_st

class TestDataset:
    # contents of test_image.py
    def test_dataset(image_file):
        img = load_image(image_file)
        # compute and test histogram