import torch
from semilearn.core.utils import get_net_builder
import semilearn.cbc_config.cbcconfig as config

if __name__ == '__main__':

    DEVICE = torch.device('cuda:%s'%(str(config.gpu_no)) if torch.cuda.is_available() else 'cpu')

    model_root_path = "/data2/code/Semi-supervised-learning/saved_models/softmatch_abc/softmatch_cbcnet_abc_100_0_adamW_cosine"
    model_path=fr"{model_root_path}/model_best.pth"
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, DEVICE)
    net_builder = get_net_builder('CBCNet', from_name=False)
    model = net_builder(num_classes=len(config.alphabet), pretrained=False)
    model.to(DEVICE)

    state_dict = checkpoint['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[7:]  # 去掉 'module.'
        else:
            new_key = k
        new_state_dict[new_key] = v

    new_state_dict1 = {}
    for k, v in new_state_dict.items():
        if k.startswith('backbone.'):
            new_key = k[9:]  # 去掉 'module.'
        elif k.startswith('aux_classifier.'):
            continue
        else:
            new_key = k
        new_state_dict1[new_key]=v

    model.load_state_dict(new_state_dict1)
    model.eval()
    example = torch.rand(1, 3, config.input_h, config.input_w).type(torch.float32).to(DEVICE)
    dynamic_axes= {
        'input_tensors' : {0 : 'batch_size'},
        'preds' : {0 : 'batch_size'},
        # 'preds_size':{ 0:'batch_size'}
    }
    torch.onnx.export(model, example,f'{model_root_path}/model_best.onnx',export_params=True,verbose=False,do_constant_folding=True,input_names = ['input_tensors'],output_names = ['preds'],dynamic_axes=dynamic_axes,opset_version=16)