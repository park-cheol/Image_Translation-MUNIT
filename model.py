import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

##################################
# Adain, LayerNorm 또는 ResidualBlock module
##################################
class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # todo weights 와 bias를 동적으로 할당
        self.weight = None
        self.bias = None
        # todo dummy buffers, not used?
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        # register_buffer: 매개변수로 간주되지 않는 버퍼를 등록하는데 사용
        # batchnorm에서 running_mean은 매개변수는 아니지만 상태로써 사용

    def forward(self, input):
        assert (self.weight is not None and self.bias is not None),
        "weibt와 bias를 지정"
        b, c, h, w = input.size()
        running_mean = self.running_mean.repeat(b) # batch 수만큼 곱
        running_var = self.running_var.repeat(b)

        # instance Norm 적용
        x_reshaped = input.contiguous().view(1, b * c, h, w) # batch: 1로
        """contiguous
        narrow(), view(), expand() and transpose() 같은 것은 텐서의 내용을 실제로 바꾸지 않고 인덱스만
        바꿔 동작을 처리하는 함수, 즉 메타정보를 수정하기 때문에 offset과 stride가 새로운 모양을 갖음
        이 원래텐서와 바뀐텐서는 메모리를 공유하고 있는 상태
        x = torch.randn(3,2)
        y = torch.transpose(x, 0, 1)
        x[0, 0] = 42
        print(y[0,0])
        # prints 42
        x는 연속적이지만 y는 레이아웃이 처음부터 같은 모양 텐서와 다르므로 아님
        continguous()를 호출하면 실제로 텐서의 복사본이 만들어지므로 요소의 순서는 같은 모양의 텐서가 처음부터 만들
        어진것처럼 같다.
        ### 다른사람 답변
        contiguous함수는 텐서를 numpy같은 방식으로 메모리에 저장하는 방식을 말합니다 python에서 list타입의 변수는 크기가 가변적이고 
        어떤 타입의 원소이든 저장할수 있지만 독립적인 메모리에 저장되있어 접근속도가 느립니다
        반면에 numpy는 윗분링크처럼 인접한 배열의 데이터는 인접한 메모리에 저장함으로써 접근속도나 transpose속도가 매우 빠르게됩니다
        보통 view함수를 써서 텐서모양을 고칠때 contiguous형식이 요구되는데 view함수는 reshape나 resize와는 다르게 어떤 경우에도 
        메모리복사없이 이루어집니다 따라서 contiguous형식이 아닐때는 텐서모양을 고칠수 없게되고 런타임에러가 발생합니다
        요약하자면 몇몇함수가 메모리 효율적인 연산을 위해 contiguous형식을 요구하니 그 함수를 사용할때만 contiguous형식으로 맞춰주면 될것같습니다
        """

        out = F.batch_norm(x_reshaped, running_mean, self.weight, self.bias, True, self.momentum, self.eps)
        # if you set training=True then batch_norm computes and uses the appropriate normalization statistics for the argued batch
        # (this means we don't need to calculate the mean and std ourselves). Your argued mu and stddev are supposed to be the running mean and running std for all training batches.
        # These tensors are updated with the new batch statistics in the batch_norm function.
        return out.view(b, c, h, w)

    def __repr__(self): # todo 좀더 자세히
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"
    # __repr__ : 시스템이 인식하는대로, 객체 모습 그대로 호출
    # __str__ : 사용자 입장에서 보기 쉬운, 간소화한


class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            # BN이나 IN, gamma에 uniform 초기화 하고 beta에는 0으로 초기화
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))
            # torch.nn.Parameter 클래스는 torch.Tensor 클래스를 상속받아 만들어졌고, torch.nn.Module 클래스의 attribute로 할당하면,
            # 자동으로 파라메터 리스트에 추가되는 것이 기존의 torch.Tensor 클래스와의 차이점이라고 한다.

    def forward(self, input): # todo input의 shape 찍어보기 아마도[B,C,H,W]
        shape = [-1] + [1] * (input.dim() - 1)
        # 만약 input = [B,C,H,W]라면 shape [-1, 1, 1, 1]
        mean = input.view(input.size(0), -1).mean(1).view(*shape) # todo 확인[B, 1, 1, 1]
        std = input.view(input.size(0), -1).std(1).view(*shape) # todo 확인[B, 1, 1, 1]
        input = (input - mean) / (std + self.eps) # 정규화

        if self.affine:
            shape = [1, -1] + [1] * (input.dim() - 2) # todo 만약 input이 [B,C,H,W] => [1, -1, 1, 1]
            input = input * self.gamma.view(*shape) + self.beta.view(*shape)
            # todo afiine은 채널과 무슨 관련이 있는지에 대해서

        return input

class ResidualBlock(nn.Module):

    def __init__(self, features, norm="in"):
        super(ResidualBlock, self).__init__()

        norm_layer = AdaptiveInstanceNorm2d if norm == "adain" else nn.InstanceNorm2d
        # content Encoder에는 IN을 쓰지만 Stlye Encoder에서는 IN을 쓰지 않는다
        # IN이 스타일 정보의 손실을 일으키므로
        # Decoder 파트 에서 adain 사용할 예정

        self.layer1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
        )

        self.layer2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        layer1_output = self.layer1(input)
        layer1_act = self.relu(layer1_output)
        layer2_output = self.layer2(layer1_act)

        return layer2_output + input

#################################
# Style, Content Encoder
#################################

class ContentEncoder(nn.Module):

    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2):
        super(ContentEncoder, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        # 초기 Conv Layer
        self.first_layer = nn.Sequential(
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(dim),
        )

        # Downsampling
        self.downsample_layer1 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 2),
        )

        self.downsample_layer2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, stride=2, padding=1),
            nn.InstanceNorm2d(dim * 4),
        )

        # Residual Blocks
        self.residual_layer1 = ResidualBlock(dim * 4, norm='in')
        self.residual_layer2 = ResidualBlock(dim * 4, norm='in')
        self.residual_layer3 = ResidualBlock(dim * 4, norm='in')

    def forward(self, input):
        first_layer = self.first_layer(input)
        first_layer_act = self.relu(first_layer)

        downsample_layer1 = self.downsample_layer1(first_layer_act)
        downsample_layer1_act = self.relu(downsample_layer1)
        downsample_layer2 = self.downsample_layer1(downsample_layer1_act)
        downsample_layer2_act = self.relu(downsample_layer2)

        residual_layer1 = self.residual_layer1(downsample_layer2_act)
        residual_layer2 = self.residual_layer1(residual_layer1)
        output = self.residual_layer1(residual_layer2)

        return output

class StyleEncoder(nn.Module):
    #todo style_dim= paer)fck: k filter 가진 FC layer = 8개의 class를 가지다는 것인대 그럼 8개의 다양한 스타일을 가지다는뜻?
    def __init__(self, in_channels=3, dim=64, n_downsample=2, style_dim=8):
        super(StyleEncoder, self).__init__()
        # todo downsample 마지막부분 생각



































