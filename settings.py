######### global settings  #########
GPU = False # running on GPU is highly suggested
TEST_MODE = False                            # turning on the testmode means the code will run on a small dataset.
CLEAN = False                               # set to "True" if you want to clean the temporary large files after generating result
CATAGORIES = ["object", "part"]             # concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"
APP = "classification"                      # "classification", "imagecap", "vqa"

QUANTILE = 0.005                            # the threshold used for activation
CAM_THRESHOLD = 0.5
SEG_THRESHOLD = 0.04                        # the threshold used for visualization
CONCEPT_TOPN = 5
FONT_PATH = "components/font.ttf"
FONT_SIZE = 9

EPOCHS = 100
SNAPSHOT_FREQ = 5
SINGLE_LABEL = True
SEG_RESOLUTION = 7


########### sub settings ###########
# In most of the case, you don't have to change them.
# DATA_DIRECTORY: where broaden dataset locates
# IMG_SIZE: image size, alexnet use 227x227
# NUM_CLASSES: how many labels in final prediction
# FEATURE_NAMES: the array of layer where features will be extracted
# MODEL_FILE: the model file to be probed, "None" means the pretrained model in torchvision
# MODEL_PARALLEL: some model is trained in multi-GPU, so there is another way to load them.
# WORKERS: how many workers are fetching images
# BATCH_SIZE: batch size used in feature extraction
# TALLY_BATCH_SIZE: batch size used in tallying
# INDEX_FILE: if you turn on the TEST_MODE, actually you should provide this file on your own


if APP == "imagecap":
    CAM = False
    CNN_MODEL = 'resnet18'
    MODEL_FILE = 'zoo/imagecap.pth.tar'
    OUTPUT_FOLDER = "result/imagecap_"+CNN_MODEL
    MAX_SENT_LEN = 20


elif APP == "vqa":
    CAM = False
    GRAD_CAM = True
    CNN_MODEL = 'resnet152'
    MODEL_FILE = 'zoo/vqa_40.pth'
    OUTPUT_FOLDER = "result/vqa_"+CNN_MODEL
    IMG_SIZE = 224
    DATA_DIRECTORY = '../NetDissect-Lite/dataset/broden1_224'
    OUTPUT_FEATURE_SIZE = 2048
    MAX_ANSWERS = 3000
    VQA_IMG_PATH = '/home/sunyiyou/dataset/coco/test_vqa/image'
    VQA_QUESTIONS_FILE = '/home/sunyiyou/dataset/coco/test_vqa/test_OpenEnded_questions.json'
    VQA_ANSWERS_FILE = '/home/sunyiyou/dataset/coco/test_vqa/test_answers.json'
    VQA_IMAGE_INDEX_FILE = '/home/sunyiyou/dataset/coco/test_vqa/test_coco_img.npy'
    VOCAB_FILE = '/home/sunyiyou/PycharmProjects/NetDissect2/components/vocab.json'

elif APP == "classification":
    CAM = False

    CNN_MODEL = 'alexnet'  # model arch: wide_resnet18, resnet18, alexnet, resnet50, densenet161
    DATASET = 'imagenet'  # model trained on: places365 or imagenet
    OUTPUT_FOLDER = "result/pytorch_"+CNN_MODEL+"_"+DATASET

    # DATASET_PATH = '/home/sunyiyou/dataset/places365_standard'
    # DATASET_INDEX_FILE = '/home/sunyiyou/dataset/places365_standard/val_sample.txt'
    DATASET_PATH = "/Users/sunyiyou/Desktop/workspace/dataset/places/val/"
    DATASET_INDEX_FILE = "/Users/sunyiyou/Desktop/workspace/dataset/places/val_sample.txt"
    if CNN_MODEL == "alexnet":
        GRAD_CAM = True
    else:
        GRAD_CAM = False
    if DATASET == 'places365':
        NUM_CLASSES = 365
        if CNN_MODEL == 'resnet18':
            MODEL_FILE = 'zoo/resnet18_places365.pth.tar'
            MODEL_PARALLEL = True
        elif CNN_MODEL == 'wideresnet18':
            MODEL_FILE = 'zoo/whole_wideresnet18_places365_python36.pth.tar'
            MODEL_PARALLEL = True
        else:
            MODEL_FILE = 'zoo/whole_'+CNN_MODEL+'_places365_python36.pth.tar'
            MODEL_PARALLEL = False
    elif DATASET == 'imagenet':
        NUM_CLASSES = 1000
        MODEL_FILE = None
        MODEL_PARALLEL = False

if APP != "vqa":
    if CNN_MODEL != 'alexnet':
        DATA_DIRECTORY = '../NetDissect-Lite/dataset/broden1_224'
        IMG_SIZE = 224
    else:
        DATA_DIRECTORY = '../NetDissect-Lite/dataset/broden1_227'
        IMG_SIZE = 227

if CNN_MODEL.startswith('resnet'):
    FEATURE_NAMES = ['layer4']
elif CNN_MODEL == 'densenet161' or CNN_MODEL == 'alexnet' or CNN_MODEL.startswith('vgg'):
    FEATURE_NAMES = ['features']


if TEST_MODE:
    WORKERS = 1
    BATCH_SIZE = 4
    FEAT_BATCH_SIZE = 16
    TALLY_BATCH_SIZE = 2
    TALLY_AHEAD = 1
    INDEX_FILE = 'index_sm.csv'
    OUTPUT_FOLDER += "_test"
else:
    WORKERS = 12
    BATCH_SIZE = 128
    FEAT_BATCH_SIZE = 16
    TALLY_BATCH_SIZE = 16
    TALLY_AHEAD = 4
    INDEX_FILE = 'index.csv'
