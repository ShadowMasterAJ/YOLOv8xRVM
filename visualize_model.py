import torch
from model.model import MattingNetwork
import torch.onnx
import netron
from torchvision import transforms
from PIL import Image

def visualize():
    model = MattingNetwork("yolov8").eval().cuda()  # or "resnet50"

    model.load_state_dict(torch.load("checkpoint/yolov8n-seg.pt"),strict=False)
    img = Image.open('test_image.jpg')
    convert_tensor = transforms.ToTensor()

    input_tensor = convert_tensor(img).cuda()
    input_tensor=input_tensor[None,:,:,:]
    print(type(input_tensor),input_tensor.shape)
    # imgShow = np.transpose(input_tensor,(1,2,0))
    # plt.imshow(imgShow)
    # plt.show()
    
    input_tensor=torch.nn.functional.pad(input_tensor, (0,640-input_tensor.shape[3]))
    print('FINAL INPUT SHAPE:',input_tensor.shape)
    # # np.save('test_image.npy',input_tensor)
    
    input_names = ['input']
    output_names = ['output']

        
    torch.onnx.export(model, input_tensor, "my_model2.onnx", opset_version=12,training=torch.onnx.TrainingMode.EVAL, verbose=True,do_constant_folding=True, input_names=input_names, output_names=output_names,
                )

    # netron.start("my_model.onnx")
visualize()
    