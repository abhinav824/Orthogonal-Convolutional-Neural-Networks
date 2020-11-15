import torch
from torchvision import transforms , models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

device = ("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load('/home/abhinav/my_projects/ocnn/Orthogonal-Convolutional-Neural-Networks/classification/model/resnet34_49')

model = models.__dict__['resnet34']()

model = torch.nn.DataParallel(model).cuda()

model.load_state_dict(checkpoint['state_dict'])

for p in model.parameters():
    p.requires_grad = False
model.to(device)

def model_activations(input,model):

    # layers = {
    # '4' : 'conv1_1',
    # '21' : 'conv2_1',
    # '35': 'conv3_1',
    # '50': 'conv4_1',
    # '64': 'conv4_2',
    # '70': 'conv5_1'
    # }
    # layers_new =  {
    #     "layer1" : model.module.layer1,
    #     "layer2" : model.module.layer2,
    #     "layer3" : model.module.layer3,
    #     "layer4" : model.module.layer4,
    # }
    # layers_list = [model.module.conv1, model.module.bn1, model.module.relu, model.module.maxpool]

    # for name,block_list in layers_new.items():
    #     for block_value in block_list:
    #         for appl_layer in block_value.children():
    #             layers_list.append(appl_layer)
    # # print(model.module)
    # # print(layers_list)
    #
    # for i in range(len(layers_list)):
    #     layer = layers_list[i](x)
    #     x = layer(x)
    #     if i in layers:
    #         features[layers[name]] = x
    features = {}

    x = input
    x = x.unsqueeze(0)

    model_module = model.module

    #applying layers
    x = model_module.conv1(x)
    features['conv1_1'] = x
    x = model_module.bn1(x)
    x = model_module.relu(x)
    x = model_module.maxpool(x)
    #layer1 block0
    x = model_module.layer1[0].conv1(x)
    features['conv2_1'] = x
    x = model_module.layer1[0].bn1(x)
    x = model_module.layer1[0].relu(x)
    x = model_module.layer1[0].conv2(x)
    x = model_module.layer1[0].bn2(x)
    #layer1 block1
    x = model_module.layer1[1].conv1(x)
    x = model_module.layer1[1].bn1(x)
    x = model_module.layer1[1].relu(x)
    x = model_module.layer1[1].conv2(x)
    x = model_module.layer1[1].bn2(x)
    #layer1 block2
    x = model_module.layer1[2].conv1(x)
    x = model_module.layer1[2].bn1(x)
    x = model_module.layer1[2].relu(x)
    x = model_module.layer1[2].conv2(x)
    x = model_module.layer1[2].bn2(x)
    #layer2 block0
    x = model_module.layer2[0].conv1(x)
    features['conv3_1'] = x
    x = model_module.layer2[0].bn1(x)
    x = model_module.layer2[0].relu(x)
    x = model_module.layer2[0].conv2(x)
    x = model_module.layer2[0].bn2(x)
    #layer2 block1
    x = model_module.layer2[1].conv1(x)
    x = model_module.layer2[1].bn1(x)
    x = model_module.layer2[1].relu(x)
    x = model_module.layer2[1].conv2(x)
    x = model_module.layer2[1].bn2(x)
    #layer2 block2
    x = model_module.layer2[2].conv1(x)
    x = model_module.layer2[2].bn1(x)
    x = model_module.layer2[2].relu(x)
    x = model_module.layer2[2].conv2(x)
    x = model_module.layer2[2].bn2(x)
    #layer2 block3
    x = model_module.layer2[3].conv1(x)
    x = model_module.layer2[3].bn1(x)
    x = model_module.layer2[3].relu(x)
    x = model_module.layer2[3].conv2(x)
    x = model_module.layer2[3].bn2(x)
    #layer3 block0
    x = model_module.layer3[0].conv1(x)
    x = model_module.layer3[0].bn1(x)
    x = model_module.layer3[0].relu(x)
    x = model_module.layer3[0].conv2(x)
    x = model_module.layer3[0].bn2(x)
    #layer3 block1
    x = model_module.layer3[1].conv1(x)
    features['conv4_1'] = x
    x = model_module.layer3[1].bn1(x)
    x = model_module.layer3[1].relu(x)
    x = model_module.layer3[1].conv2(x)
    features['conv4_2'] = x
    x = model_module.layer3[1].bn2(x)
    #layer3 block2
    x = model_module.layer3[2].conv1(x)
    x = model_module.layer3[2].bn1(x)
    x = model_module.layer3[2].relu(x)
    x = model_module.layer3[2].conv2(x)
    x = model_module.layer3[2].bn2(x)
    #layer3 block3
    x = model_module.layer3[3].conv1(x)
    x = model_module.layer3[3].bn1(x)
    x = model_module.layer3[3].relu(x)
    x = model_module.layer3[3].conv2(x)
    x = model_module.layer3[3].bn2(x)
    #layer3 block4
    x = model_module.layer3[4].conv1(x)
    x = model_module.layer3[4].bn1(x)
    x = model_module.layer3[4].relu(x)
    x = model_module.layer3[4].conv2(x)
    x = model_module.layer3[4].bn2(x)
    #layer3 block5
    x = model_module.layer3[5].conv1(x)
    x = model_module.layer3[5].bn1(x)
    x = model_module.layer3[5].relu(x)
    x = model_module.layer3[5].conv2(x)
    x = model_module.layer3[5].bn2(x)
    #layer4 block0
    x = model_module.layer4[0].conv1(x)
    features['conv5_1'] = x
    x = model_module.layer4[0].bn1(x)
    x = model_module.layer4[0].relu(x)
    x = model_module.layer4[0].conv2(x)
    x = model_module.layer4[0].bn2(x)
    #layer4 block1
    x = model_module.layer4[1].conv1(x)
    x = model_module.layer4[1].bn1(x)
    x = model_module.layer4[1].relu(x)
    x = model_module.layer4[1].conv2(x)
    x = model_module.layer4[1].bn2(x)
    #layer4 block2
    x = model_module.layer4[2].conv1(x)
    x = model_module.layer4[2].bn1(x)
    x = model_module.layer4[2].relu(x)
    x = model_module.layer4[2].conv2(x)
    x = model_module.layer4[2].bn2(x)
    #avgpool
    x = model_module.avgpool(x)
    #fc
    # x = model_module.fc(x)

    return features

transform = transforms.Compose([transforms.Resize(300),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


content = Image.open("content.jpg").convert("RGB")
content = transform(content).to(device)
print("COntent shape => ", content.shape)
style = Image.open("style.jpg").convert("RGB")
style = transform(style).to(device)

def imcnvt(image):
    x = image.to("cpu").clone().detach().numpy().squeeze()
    x = x.transpose(1,2,0)
    x = x*np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5))
    return np.clip(x,0,1)

fig, (ax1,ax2) = plt.subplots(1,2)

ax1.imshow(imcnvt(content),label = "Content")
ax2.imshow(imcnvt(style),label = "Style")
plt.show()

def gram_matrix(imgfeature):
    _,d,h,w = imgfeature.size()
    imgfeature = imgfeature.view(d,h*w)
    gram_mat = torch.mm(imgfeature,imgfeature.t())

    return gram_mat


target = content.clone().requires_grad_(True).to(device)

#set device to cuda if available
print("device = ",device)


style_features = model_activations(style,model)
content_features = model_activations(content,model)

style_wt_meas = {"conv1_1" : 1.0,
                 "conv2_1" : 0.8,
                 "conv3_1" : 0.4,
                 "conv4_1" : 0.2,
                 "conv5_1" : 0.1}

style_grams = {layer:gram_matrix(style_features[layer]) for layer in style_features}

content_wt = 100
style_wt = 1e8

print_after = 500
epochs = 2000
optimizer = torch.optim.Adam([target],lr=0.007)

for i in range(1,epochs+1):
    target_features = model_activations(target,model)
    content_loss = torch.mean((content_features['conv4_2']-target_features['conv4_2'])**2)

    style_loss = 0
    for layer in style_wt_meas:
        style_gram = style_grams[layer]
        target_gram = target_features[layer]
        _,d,w,h = target_gram.shape
        target_gram = gram_matrix(target_gram)

        style_loss += (style_wt_meas[layer]*torch.mean((target_gram-style_gram)**2))/d*w*h

    total_loss = content_wt*content_loss + style_wt*style_loss

    if i%10==0:
        print("epoch ",i," ", total_loss)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if i%print_after == 0:
        plt.imshow(imcnvt(target),label="Epoch "+str(i))
        plt.show()
        plt.imsave(str(i)+'.png',imcnvt(target),format='png')
print(total_loss)
