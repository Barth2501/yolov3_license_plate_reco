��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02unknown8��
�
cnn_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_namecnn_conv_1/kernel

%cnn_conv_1/kernel/Read/ReadVariableOpReadVariableOpcnn_conv_1/kernel*&
_output_shapes
:
*
dtype0
v
cnn_conv_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namecnn_conv_1/bias
o
#cnn_conv_1/bias/Read/ReadVariableOpReadVariableOpcnn_conv_1/bias*
_output_shapes
:
*
dtype0
�
cnn_conv_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_namecnn_conv_2/kernel

%cnn_conv_2/kernel/Read/ReadVariableOpReadVariableOpcnn_conv_2/kernel*&
_output_shapes
:
*
dtype0
v
cnn_conv_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namecnn_conv_2/bias
o
#cnn_conv_2/bias/Read/ReadVariableOpReadVariableOpcnn_conv_2/bias*
_output_shapes
:*
dtype0
~
cnn_dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*!
shared_namecnn_dense/kernel
w
$cnn_dense/kernel/Read/ReadVariableOpReadVariableOpcnn_dense/kernel* 
_output_shapes
:
��*
dtype0
u
cnn_dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_namecnn_dense/bias
n
"cnn_dense/bias/Read/ReadVariableOpReadVariableOpcnn_dense/bias*
_output_shapes	
:�*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�
*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	�
*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:
*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
input_layer
conv_layer_1
pool_layer_1
conv_layer_2
pool_layer_2
flatten
dense_layer
output_layer
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	
signatures

trainable_variables
regularization_losses
	variables
	keras_api
 
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
 	variables
!	keras_api
R
"trainable_variables
#regularization_losses
$	variables
%	keras_api
h

&kernel
'bias
(trainable_variables
)regularization_losses
*	variables
+	keras_api
h

,kernel
-bias
.trainable_variables
/regularization_losses
0	variables
1	keras_api
 
8
0
1
2
3
&4
'5
,6
-7
 
8
0
1
2
3
&4
'5
,6
-7
�
2metrics
3layer_regularization_losses
4non_trainable_variables

trainable_variables
regularization_losses
	variables

5layers
US
VARIABLE_VALUEcnn_conv_1/kernel.conv_layer_1/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcnn_conv_1/bias,conv_layer_1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
6metrics
7layer_regularization_losses
8non_trainable_variables
trainable_variables
regularization_losses
	variables

9layers
 
 
 
�
:metrics
;layer_regularization_losses
<non_trainable_variables
trainable_variables
regularization_losses
	variables

=layers
US
VARIABLE_VALUEcnn_conv_2/kernel.conv_layer_2/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcnn_conv_2/bias,conv_layer_2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
>metrics
?layer_regularization_losses
@non_trainable_variables
trainable_variables
regularization_losses
	variables

Alayers
 
 
 
�
Bmetrics
Clayer_regularization_losses
Dnon_trainable_variables
trainable_variables
regularization_losses
 	variables

Elayers
 
 
 
�
Fmetrics
Glayer_regularization_losses
Hnon_trainable_variables
"trainable_variables
#regularization_losses
$	variables

Ilayers
SQ
VARIABLE_VALUEcnn_dense/kernel-dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcnn_dense/bias+dense_layer/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1
 

&0
'1
�
Jmetrics
Klayer_regularization_losses
Lnon_trainable_variables
(trainable_variables
)regularization_losses
*	variables

Mlayers
PN
VARIABLE_VALUEdense/kernel.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUE
dense/bias,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1
 

,0
-1
�
Nmetrics
Olayer_regularization_losses
Pnon_trainable_variables
.trainable_variables
/regularization_losses
0	variables

Qlayers
 
 
 
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
�
serving_default_input_1Placeholder*/
_output_shapes
:���������*
dtype0*$
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1cnn_conv_1/kernelcnn_conv_1/biascnn_conv_2/kernelcnn_conv_2/biascnn_dense/kernelcnn_dense/biasdense/kernel
dense/bias*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*5
config_proto%#

CPU

GPU2 *0J 8R(*+
f&R$
"__inference_signature_wrapper_5677
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%cnn_conv_1/kernel/Read/ReadVariableOp#cnn_conv_1/bias/Read/ReadVariableOp%cnn_conv_2/kernel/Read/ReadVariableOp#cnn_conv_2/bias/Read/ReadVariableOp$cnn_dense/kernel/Read/ReadVariableOp"cnn_dense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *5
config_proto%#

CPU

GPU2 *0J 8R(*&
f!R
__inference__traced_save_5725
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecnn_conv_1/kernelcnn_conv_1/biascnn_conv_2/kernelcnn_conv_2/biascnn_dense/kernelcnn_dense/biasdense/kernel
dense/bias*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *5
config_proto%#

CPU

GPU2 *0J 8R(*)
f$R"
 __inference__traced_restore_5761��
�	
�
A__inference_cnn_dense_layer_call_and_return_conditional_losses_22

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�!
�
<__inference_cnn_layer_call_and_return_conditional_losses_329

inputs-
)cnn_conv_1_statefulpartitionedcall_args_1-
)cnn_conv_1_statefulpartitionedcall_args_2-
)cnn_conv_2_statefulpartitionedcall_args_1-
)cnn_conv_2_statefulpartitionedcall_args_2,
(cnn_dense_statefulpartitionedcall_args_1,
(cnn_dense_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity��"cnn_conv_1/StatefulPartitionedCall�"cnn_conv_2/StatefulPartitionedCall�!cnn_dense/StatefulPartitionedCall�dense/StatefulPartitionedCall�
"cnn_conv_1/StatefulPartitionedCallStatefulPartitionedCallinputs)cnn_conv_1_statefulpartitionedcall_args_1)cnn_conv_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_cnn_conv_1_layer_call_and_return_conditional_losses_1442$
"cnn_conv_1/StatefulPartitionedCall�
cnn_maxpool_1/PartitionedCallPartitionedCall+cnn_conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_cnn_maxpool_1_layer_call_and_return_conditional_losses_1272
cnn_maxpool_1/PartitionedCall�
"cnn_conv_2/StatefulPartitionedCallStatefulPartitionedCall&cnn_maxpool_1/PartitionedCall:output:0)cnn_conv_2_statefulpartitionedcall_args_1)cnn_conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_cnn_conv_2_layer_call_and_return_conditional_losses_452$
"cnn_conv_2/StatefulPartitionedCall�
cnn_maxpool_2/PartitionedCallPartitionedCall+cnn_conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_cnn_maxpool_2_layer_call_and_return_conditional_losses_62
cnn_maxpool_2/PartitionedCall�
cnn_flatten/PartitionedCallPartitionedCall&cnn_maxpool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_cnn_flatten_layer_call_and_return_conditional_losses_3102
cnn_flatten/PartitionedCall�
!cnn_dense/StatefulPartitionedCallStatefulPartitionedCall$cnn_flatten/PartitionedCall:output:0(cnn_dense_statefulpartitionedcall_args_1(cnn_dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_cnn_dense_layer_call_and_return_conditional_losses_2482#
!cnn_dense/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall*cnn_dense/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_1622
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0#^cnn_conv_1/StatefulPartitionedCall#^cnn_conv_2/StatefulPartitionedCall"^cnn_dense/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::2H
"cnn_conv_1/StatefulPartitionedCall"cnn_conv_1/StatefulPartitionedCall2H
"cnn_conv_2/StatefulPartitionedCall"cnn_conv_2/StatefulPartitionedCall2F
!cnn_dense/StatefulPartitionedCall!cnn_dense/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�

�
!__inference_cnn_layer_call_fn_355

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*E
f@R>
<__inference_cnn_layer_call_and_return_conditional_losses_3292
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�-
�
<__inference_cnn_layer_call_and_return_conditional_losses_122

inputs-
)cnn_conv_1_conv2d_readvariableop_resource.
*cnn_conv_1_biasadd_readvariableop_resource-
)cnn_conv_2_conv2d_readvariableop_resource.
*cnn_conv_2_biasadd_readvariableop_resource,
(cnn_dense_matmul_readvariableop_resource-
)cnn_dense_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity��!cnn_conv_1/BiasAdd/ReadVariableOp� cnn_conv_1/Conv2D/ReadVariableOp�!cnn_conv_2/BiasAdd/ReadVariableOp� cnn_conv_2/Conv2D/ReadVariableOp� cnn_dense/BiasAdd/ReadVariableOp�cnn_dense/MatMul/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�
 cnn_conv_1/Conv2D/ReadVariableOpReadVariableOp)cnn_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02"
 cnn_conv_1/Conv2D/ReadVariableOp�
cnn_conv_1/Conv2DConv2Dinputs(cnn_conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2
cnn_conv_1/Conv2D�
!cnn_conv_1/BiasAdd/ReadVariableOpReadVariableOp*cnn_conv_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02#
!cnn_conv_1/BiasAdd/ReadVariableOp�
cnn_conv_1/BiasAddBiasAddcnn_conv_1/Conv2D:output:0)cnn_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
2
cnn_conv_1/BiasAdd�
cnn_conv_1/ReluRelucnn_conv_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������
2
cnn_conv_1/Relu�
cnn_maxpool_1/MaxPoolMaxPoolcnn_conv_1/Relu:activations:0*/
_output_shapes
:���������
*
ksize
*
paddingVALID*
strides
2
cnn_maxpool_1/MaxPool�
 cnn_conv_2/Conv2D/ReadVariableOpReadVariableOp)cnn_conv_2_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02"
 cnn_conv_2/Conv2D/ReadVariableOp�
cnn_conv_2/Conv2DConv2Dcnn_maxpool_1/MaxPool:output:0(cnn_conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
cnn_conv_2/Conv2D�
!cnn_conv_2/BiasAdd/ReadVariableOpReadVariableOp*cnn_conv_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!cnn_conv_2/BiasAdd/ReadVariableOp�
cnn_conv_2/BiasAddBiasAddcnn_conv_2/Conv2D:output:0)cnn_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
cnn_conv_2/BiasAdd�
cnn_conv_2/ReluRelucnn_conv_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
cnn_conv_2/Relu�
cnn_maxpool_2/MaxPoolMaxPoolcnn_conv_2/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
cnn_maxpool_2/MaxPoolw
cnn_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
cnn_flatten/Const�
cnn_flatten/ReshapeReshapecnn_maxpool_2/MaxPool:output:0cnn_flatten/Const:output:0*
T0*(
_output_shapes
:����������2
cnn_flatten/Reshape�
cnn_dense/MatMul/ReadVariableOpReadVariableOp(cnn_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02!
cnn_dense/MatMul/ReadVariableOp�
cnn_dense/MatMulMatMulcnn_flatten/Reshape:output:0'cnn_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
cnn_dense/MatMul�
 cnn_dense/BiasAdd/ReadVariableOpReadVariableOp)cnn_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 cnn_dense/BiasAdd/ReadVariableOp�
cnn_dense/BiasAddBiasAddcnn_dense/MatMul:product:0(cnn_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
cnn_dense/BiasAddw
cnn_dense/ReluRelucnn_dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
cnn_dense/Relu�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulcnn_dense/Relu:activations:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense/BiasAdds
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense/Softmax�
IdentityIdentitydense/Softmax:softmax:0"^cnn_conv_1/BiasAdd/ReadVariableOp!^cnn_conv_1/Conv2D/ReadVariableOp"^cnn_conv_2/BiasAdd/ReadVariableOp!^cnn_conv_2/Conv2D/ReadVariableOp!^cnn_dense/BiasAdd/ReadVariableOp ^cnn_dense/MatMul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::2F
!cnn_conv_1/BiasAdd/ReadVariableOp!cnn_conv_1/BiasAdd/ReadVariableOp2D
 cnn_conv_1/Conv2D/ReadVariableOp cnn_conv_1/Conv2D/ReadVariableOp2F
!cnn_conv_2/BiasAdd/ReadVariableOp!cnn_conv_2/BiasAdd/ReadVariableOp2D
 cnn_conv_2/Conv2D/ReadVariableOp cnn_conv_2/Conv2D/ReadVariableOp2D
 cnn_dense/BiasAdd/ReadVariableOp cnn_dense/BiasAdd/ReadVariableOp2B
cnn_dense/MatMul/ReadVariableOpcnn_dense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
'__inference_cnn_dense_layer_call_fn_255

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_cnn_dense_layer_call_and_return_conditional_losses_2482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
(__inference_cnn_conv_1_layer_call_fn_151

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_cnn_conv_1_layer_call_and_return_conditional_losses_1442
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������
2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
G
+__inference_cnn_maxpool_1_layer_call_fn_132

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4������������������������������������**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_cnn_maxpool_1_layer_call_and_return_conditional_losses_1272
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
b
F__inference_cnn_maxpool_1_layer_call_and_return_conditional_losses_127

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�-
�
<__inference_cnn_layer_call_and_return_conditional_losses_205

inputs-
)cnn_conv_1_conv2d_readvariableop_resource.
*cnn_conv_1_biasadd_readvariableop_resource-
)cnn_conv_2_conv2d_readvariableop_resource.
*cnn_conv_2_biasadd_readvariableop_resource,
(cnn_dense_matmul_readvariableop_resource-
)cnn_dense_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity��!cnn_conv_1/BiasAdd/ReadVariableOp� cnn_conv_1/Conv2D/ReadVariableOp�!cnn_conv_2/BiasAdd/ReadVariableOp� cnn_conv_2/Conv2D/ReadVariableOp� cnn_dense/BiasAdd/ReadVariableOp�cnn_dense/MatMul/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�
 cnn_conv_1/Conv2D/ReadVariableOpReadVariableOp)cnn_conv_1_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02"
 cnn_conv_1/Conv2D/ReadVariableOp�
cnn_conv_1/Conv2DConv2Dinputs(cnn_conv_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
*
paddingVALID*
strides
2
cnn_conv_1/Conv2D�
!cnn_conv_1/BiasAdd/ReadVariableOpReadVariableOp*cnn_conv_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02#
!cnn_conv_1/BiasAdd/ReadVariableOp�
cnn_conv_1/BiasAddBiasAddcnn_conv_1/Conv2D:output:0)cnn_conv_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������
2
cnn_conv_1/BiasAdd�
cnn_conv_1/ReluRelucnn_conv_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������
2
cnn_conv_1/Relu�
cnn_maxpool_1/MaxPoolMaxPoolcnn_conv_1/Relu:activations:0*/
_output_shapes
:���������
*
ksize
*
paddingVALID*
strides
2
cnn_maxpool_1/MaxPool�
 cnn_conv_2/Conv2D/ReadVariableOpReadVariableOp)cnn_conv_2_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02"
 cnn_conv_2/Conv2D/ReadVariableOp�
cnn_conv_2/Conv2DConv2Dcnn_maxpool_1/MaxPool:output:0(cnn_conv_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingVALID*
strides
2
cnn_conv_2/Conv2D�
!cnn_conv_2/BiasAdd/ReadVariableOpReadVariableOp*cnn_conv_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!cnn_conv_2/BiasAdd/ReadVariableOp�
cnn_conv_2/BiasAddBiasAddcnn_conv_2/Conv2D:output:0)cnn_conv_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
cnn_conv_2/BiasAdd�
cnn_conv_2/ReluRelucnn_conv_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
cnn_conv_2/Relu�
cnn_maxpool_2/MaxPoolMaxPoolcnn_conv_2/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingVALID*
strides
2
cnn_maxpool_2/MaxPoolw
cnn_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
cnn_flatten/Const�
cnn_flatten/ReshapeReshapecnn_maxpool_2/MaxPool:output:0cnn_flatten/Const:output:0*
T0*(
_output_shapes
:����������2
cnn_flatten/Reshape�
cnn_dense/MatMul/ReadVariableOpReadVariableOp(cnn_dense_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02!
cnn_dense/MatMul/ReadVariableOp�
cnn_dense/MatMulMatMulcnn_flatten/Reshape:output:0'cnn_dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
cnn_dense/MatMul�
 cnn_dense/BiasAdd/ReadVariableOpReadVariableOp)cnn_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 cnn_dense/BiasAdd/ReadVariableOp�
cnn_dense/BiasAddBiasAddcnn_dense/MatMul:product:0(cnn_dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
cnn_dense/BiasAddw
cnn_dense/ReluRelucnn_dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
cnn_dense/Relu�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulcnn_dense/Relu:activations:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
dense/BiasAdds
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
dense/Softmax�
IdentityIdentitydense/Softmax:softmax:0"^cnn_conv_1/BiasAdd/ReadVariableOp!^cnn_conv_1/Conv2D/ReadVariableOp"^cnn_conv_2/BiasAdd/ReadVariableOp!^cnn_conv_2/Conv2D/ReadVariableOp!^cnn_dense/BiasAdd/ReadVariableOp ^cnn_dense/MatMul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::2F
!cnn_conv_1/BiasAdd/ReadVariableOp!cnn_conv_1/BiasAdd/ReadVariableOp2D
 cnn_conv_1/Conv2D/ReadVariableOp cnn_conv_1/Conv2D/ReadVariableOp2F
!cnn_conv_2/BiasAdd/ReadVariableOp!cnn_conv_2/BiasAdd/ReadVariableOp2D
 cnn_conv_2/Conv2D/ReadVariableOp cnn_conv_2/Conv2D/ReadVariableOp2D
 cnn_dense/BiasAdd/ReadVariableOp cnn_dense/BiasAdd/ReadVariableOp2B
cnn_dense/MatMul/ReadVariableOpcnn_dense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�

�
!__inference_cnn_layer_call_fn_430
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*E
f@R>
<__inference_cnn_layer_call_and_return_conditional_losses_4172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
� 
�
__inference__traced_save_5725
file_prefix0
,savev2_cnn_conv_1_kernel_read_readvariableop.
*savev2_cnn_conv_1_bias_read_readvariableop0
,savev2_cnn_conv_2_kernel_read_readvariableop.
*savev2_cnn_conv_2_bias_read_readvariableop/
+savev2_cnn_dense_kernel_read_readvariableop-
)savev2_cnn_dense_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b144d4a1d3444e6284b6a6e5af845458/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B.conv_layer_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB,conv_layer_1/bias/.ATTRIBUTES/VARIABLE_VALUEB.conv_layer_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB,conv_layer_2/bias/.ATTRIBUTES/VARIABLE_VALUEB-dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB+dense_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_cnn_conv_1_kernel_read_readvariableop*savev2_cnn_conv_1_bias_read_readvariableop,savev2_cnn_conv_2_kernel_read_readvariableop*savev2_cnn_conv_2_bias_read_readvariableop+savev2_cnn_dense_kernel_read_readvariableop)savev2_cnn_dense_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes

22
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*k
_input_shapesZ
X: :
:
:
::
��:�:	�
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
�	
�
B__inference_cnn_dense_layer_call_and_return_conditional_losses_248

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
F
*__inference_cnn_maxpool_2_layer_call_fn_11

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4������������������������������������**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_cnn_maxpool_2_layer_call_and_return_conditional_losses_62
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�	
�
=__inference_dense_layer_call_and_return_conditional_losses_33

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�
`
D__inference_cnn_maxpool_2_layer_call_and_return_conditional_losses_6

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
`
D__inference_cnn_flatten_layer_call_and_return_conditional_losses_211

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�

�
!__inference_cnn_layer_call_fn_342
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*E
f@R>
<__inference_cnn_layer_call_and_return_conditional_losses_3292
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
�'
�
 __inference__traced_restore_5761
file_prefix&
"assignvariableop_cnn_conv_1_kernel&
"assignvariableop_1_cnn_conv_1_bias(
$assignvariableop_2_cnn_conv_2_kernel&
"assignvariableop_3_cnn_conv_2_bias'
#assignvariableop_4_cnn_dense_kernel%
!assignvariableop_5_cnn_dense_bias#
assignvariableop_6_dense_kernel!
assignvariableop_7_dense_bias

identity_9��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B.conv_layer_1/kernel/.ATTRIBUTES/VARIABLE_VALUEB,conv_layer_1/bias/.ATTRIBUTES/VARIABLE_VALUEB.conv_layer_2/kernel/.ATTRIBUTES/VARIABLE_VALUEB,conv_layer_2/bias/.ATTRIBUTES/VARIABLE_VALUEB-dense_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB+dense_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB.output_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB,output_layer/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp"assignvariableop_cnn_conv_1_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp"assignvariableop_1_cnn_conv_1_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp$assignvariableop_2_cnn_conv_2_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp"assignvariableop_3_cnn_conv_2_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp#assignvariableop_4_cnn_dense_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp!assignvariableop_5_cnn_dense_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8�

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
�

�
__inference__wrapped_model_5663
input_1&
"cnn_statefulpartitionedcall_args_1&
"cnn_statefulpartitionedcall_args_2&
"cnn_statefulpartitionedcall_args_3&
"cnn_statefulpartitionedcall_args_4&
"cnn_statefulpartitionedcall_args_5&
"cnn_statefulpartitionedcall_args_6&
"cnn_statefulpartitionedcall_args_7&
"cnn_statefulpartitionedcall_args_8
identity��cnn/StatefulPartitionedCall�
cnn/StatefulPartitionedCallStatefulPartitionedCallinput_1"cnn_statefulpartitionedcall_args_1"cnn_statefulpartitionedcall_args_2"cnn_statefulpartitionedcall_args_3"cnn_statefulpartitionedcall_args_4"cnn_statefulpartitionedcall_args_5"cnn_statefulpartitionedcall_args_6"cnn_statefulpartitionedcall_args_7"cnn_statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*5
config_proto%#

CPU

GPU2 *0J 8R(*/
f*R(
&__inference_restored_function_body_6712
cnn/StatefulPartitionedCall�
IdentityIdentity$cnn/StatefulPartitionedCall:output:0^cnn/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::2:
cnn/StatefulPartitionedCallcnn/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
�

�
!__inference_cnn_layer_call_fn_443

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*E
f@R>
<__inference_cnn_layer_call_and_return_conditional_losses_4172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
E
)__inference_cnn_flatten_layer_call_fn_360

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_cnn_flatten_layer_call_and_return_conditional_losses_3102
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�
�
B__inference_cnn_conv_2_layer_call_and_return_conditional_losses_45

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�	
�
>__inference_dense_layer_call_and_return_conditional_losses_162

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������
2	
Softmax�
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
�!
�
<__inference_cnn_layer_call_and_return_conditional_losses_417

inputs-
)cnn_conv_1_statefulpartitionedcall_args_1-
)cnn_conv_1_statefulpartitionedcall_args_2-
)cnn_conv_2_statefulpartitionedcall_args_1-
)cnn_conv_2_statefulpartitionedcall_args_2,
(cnn_dense_statefulpartitionedcall_args_1,
(cnn_dense_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity��"cnn_conv_1/StatefulPartitionedCall�"cnn_conv_2/StatefulPartitionedCall�!cnn_dense/StatefulPartitionedCall�dense/StatefulPartitionedCall�
"cnn_conv_1/StatefulPartitionedCallStatefulPartitionedCallinputs)cnn_conv_1_statefulpartitionedcall_args_1)cnn_conv_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_cnn_conv_1_layer_call_and_return_conditional_losses_1442$
"cnn_conv_1/StatefulPartitionedCall�
cnn_maxpool_1/PartitionedCallPartitionedCall+cnn_conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_cnn_maxpool_1_layer_call_and_return_conditional_losses_1272
cnn_maxpool_1/PartitionedCall�
"cnn_conv_2/StatefulPartitionedCallStatefulPartitionedCall&cnn_maxpool_1/PartitionedCall:output:0)cnn_conv_2_statefulpartitionedcall_args_1)cnn_conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_cnn_conv_2_layer_call_and_return_conditional_losses_452$
"cnn_conv_2/StatefulPartitionedCall�
cnn_maxpool_2/PartitionedCallPartitionedCall+cnn_conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_cnn_maxpool_2_layer_call_and_return_conditional_losses_62
cnn_maxpool_2/PartitionedCall�
cnn_flatten/PartitionedCallPartitionedCall&cnn_maxpool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_cnn_flatten_layer_call_and_return_conditional_losses_3102
cnn_flatten/PartitionedCall�
!cnn_dense/StatefulPartitionedCallStatefulPartitionedCall$cnn_flatten/PartitionedCall:output:0(cnn_dense_statefulpartitionedcall_args_1(cnn_dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_cnn_dense_layer_call_and_return_conditional_losses_2482#
!cnn_dense/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall*cnn_dense/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_1622
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0#^cnn_conv_1/StatefulPartitionedCall#^cnn_conv_2/StatefulPartitionedCall"^cnn_dense/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::2H
"cnn_conv_1/StatefulPartitionedCall"cnn_conv_1/StatefulPartitionedCall2H
"cnn_conv_2/StatefulPartitionedCall"cnn_conv_2/StatefulPartitionedCall2F
!cnn_dense/StatefulPartitionedCall!cnn_dense/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
'__inference_cnn_conv_2_layer_call_fn_52

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_cnn_conv_2_layer_call_and_return_conditional_losses_452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������
::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
#__inference_dense_layer_call_fn_169

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_1622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*/
_input_shapes
:����������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�

�
&__inference_restored_function_body_671

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*5
config_proto%#

CPU

GPU2 *0J 8R(*E
f@R>
<__inference_cnn_layer_call_and_return_conditional_losses_3792
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
C__inference_cnn_conv_1_layer_call_and_return_conditional_losses_144

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������
*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������
2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������
2
Relu�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������
2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�

�
"__inference_signature_wrapper_5677
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
*5
config_proto%#

CPU

GPU2 *0J 8R(*(
f#R!
__inference__wrapped_model_56632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
�
`
D__inference_cnn_flatten_layer_call_and_return_conditional_losses_310

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:& "
 
_user_specified_nameinputs
�!
�
<__inference_cnn_layer_call_and_return_conditional_losses_379
input_1-
)cnn_conv_1_statefulpartitionedcall_args_1-
)cnn_conv_1_statefulpartitionedcall_args_2-
)cnn_conv_2_statefulpartitionedcall_args_1-
)cnn_conv_2_statefulpartitionedcall_args_2,
(cnn_dense_statefulpartitionedcall_args_1,
(cnn_dense_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity��"cnn_conv_1/StatefulPartitionedCall�"cnn_conv_2/StatefulPartitionedCall�!cnn_dense/StatefulPartitionedCall�dense/StatefulPartitionedCall�
"cnn_conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_1)cnn_conv_1_statefulpartitionedcall_args_1)cnn_conv_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_cnn_conv_1_layer_call_and_return_conditional_losses_1442$
"cnn_conv_1/StatefulPartitionedCall�
cnn_maxpool_1/PartitionedCallPartitionedCall+cnn_conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_cnn_maxpool_1_layer_call_and_return_conditional_losses_1272
cnn_maxpool_1/PartitionedCall�
"cnn_conv_2/StatefulPartitionedCallStatefulPartitionedCall&cnn_maxpool_1/PartitionedCall:output:0)cnn_conv_2_statefulpartitionedcall_args_1)cnn_conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_cnn_conv_2_layer_call_and_return_conditional_losses_452$
"cnn_conv_2/StatefulPartitionedCall�
cnn_maxpool_2/PartitionedCallPartitionedCall+cnn_conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_cnn_maxpool_2_layer_call_and_return_conditional_losses_62
cnn_maxpool_2/PartitionedCall�
cnn_flatten/PartitionedCallPartitionedCall&cnn_maxpool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_cnn_flatten_layer_call_and_return_conditional_losses_3102
cnn_flatten/PartitionedCall�
!cnn_dense/StatefulPartitionedCallStatefulPartitionedCall$cnn_flatten/PartitionedCall:output:0(cnn_dense_statefulpartitionedcall_args_1(cnn_dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_cnn_dense_layer_call_and_return_conditional_losses_2482#
!cnn_dense/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall*cnn_dense/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_1622
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0#^cnn_conv_1/StatefulPartitionedCall#^cnn_conv_2/StatefulPartitionedCall"^cnn_dense/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::2H
"cnn_conv_1/StatefulPartitionedCall"cnn_conv_1/StatefulPartitionedCall2H
"cnn_conv_2/StatefulPartitionedCall"cnn_conv_2/StatefulPartitionedCall2F
!cnn_dense/StatefulPartitionedCall!cnn_dense/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
�!
�
<__inference_cnn_layer_call_and_return_conditional_losses_398
input_1-
)cnn_conv_1_statefulpartitionedcall_args_1-
)cnn_conv_1_statefulpartitionedcall_args_2-
)cnn_conv_2_statefulpartitionedcall_args_1-
)cnn_conv_2_statefulpartitionedcall_args_2,
(cnn_dense_statefulpartitionedcall_args_1,
(cnn_dense_statefulpartitionedcall_args_2(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identity��"cnn_conv_1/StatefulPartitionedCall�"cnn_conv_2/StatefulPartitionedCall�!cnn_dense/StatefulPartitionedCall�dense/StatefulPartitionedCall�
"cnn_conv_1/StatefulPartitionedCallStatefulPartitionedCallinput_1)cnn_conv_1_statefulpartitionedcall_args_1)cnn_conv_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*L
fGRE
C__inference_cnn_conv_1_layer_call_and_return_conditional_losses_1442$
"cnn_conv_1/StatefulPartitionedCall�
cnn_maxpool_1/PartitionedCallPartitionedCall+cnn_conv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*O
fJRH
F__inference_cnn_maxpool_1_layer_call_and_return_conditional_losses_1272
cnn_maxpool_1/PartitionedCall�
"cnn_conv_2/StatefulPartitionedCallStatefulPartitionedCall&cnn_maxpool_1/PartitionedCall:output:0)cnn_conv_2_statefulpartitionedcall_args_1)cnn_conv_2_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_cnn_conv_2_layer_call_and_return_conditional_losses_452$
"cnn_conv_2/StatefulPartitionedCall�
cnn_maxpool_2/PartitionedCallPartitionedCall+cnn_conv_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_cnn_maxpool_2_layer_call_and_return_conditional_losses_62
cnn_maxpool_2/PartitionedCall�
cnn_flatten/PartitionedCallPartitionedCall&cnn_maxpool_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*M
fHRF
D__inference_cnn_flatten_layer_call_and_return_conditional_losses_3102
cnn_flatten/PartitionedCall�
!cnn_dense/StatefulPartitionedCallStatefulPartitionedCall$cnn_flatten/PartitionedCall:output:0(cnn_dense_statefulpartitionedcall_args_1(cnn_dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*(
_output_shapes
:����������**
config_proto

CPU

GPU 2J 8*K
fFRD
B__inference_cnn_dense_layer_call_and_return_conditional_losses_2482#
!cnn_dense/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCall*cnn_dense/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:���������
**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_1622
dense/StatefulPartitionedCall�
IdentityIdentity&dense/StatefulPartitionedCall:output:0#^cnn_conv_1/StatefulPartitionedCall#^cnn_conv_2/StatefulPartitionedCall"^cnn_dense/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::2H
"cnn_conv_1/StatefulPartitionedCall"cnn_conv_1/StatefulPartitionedCall2H
"cnn_conv_2/StatefulPartitionedCall"cnn_conv_2/StatefulPartitionedCall2F
!cnn_dense/StatefulPartitionedCall!cnn_dense/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:' #
!
_user_specified_name	input_1"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_18
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������
tensorflow/serving/predict:��
�
input_layer
conv_layer_1
pool_layer_1
conv_layer_2
pool_layer_2
flatten
dense_layer
output_layer
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	
signatures

trainable_variables
regularization_losses
	variables
	keras_api
R_default_save_signature
S__call__
*T&call_and_return_all_conditional_losses"�
_tf_keras_model�{"class_name": "CNN", "name": "cnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "CNN"}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "cnn_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 28, 28, 1], "config": {"batch_input_shape": [null, 28, 28, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "cnn_input"}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
U__call__
*V&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "cnn_conv_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "cnn_conv_1", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 1}}}}
�
trainable_variables
regularization_losses
	variables
	keras_api
W__call__
*X&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "cnn_maxpool_1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "cnn_maxpool_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "cnn_conv_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "cnn_conv_2", "trainable": true, "dtype": "float32", "filters": 20, "kernel_size": [5, 5], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 10}}}}
�
trainable_variables
regularization_losses
 	variables
!	keras_api
[__call__
*\&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "cnn_maxpool_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "cnn_maxpool_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
"trainable_variables
#regularization_losses
$	variables
%	keras_api
]__call__
*^&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Flatten", "name": "cnn_flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "cnn_flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
�

&kernel
'bias
(trainable_variables
)regularization_losses
*	variables
+	keras_api
___call__
*`&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "cnn_dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "cnn_dense", "trainable": true, "dtype": "float32", "units": 500, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 320}}}}
�

,kernel
-bias
.trainable_variables
/regularization_losses
0	variables
1	keras_api
a__call__
*b&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 500}}}}
,
cserving_default"
signature_map
X
0
1
2
3
&4
'5
,6
-7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
&4
'5
,6
-7"
trackable_list_wrapper
�
2metrics
3layer_regularization_losses
4non_trainable_variables

trainable_variables
regularization_losses
	variables

5layers
S__call__
R_default_save_signature
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
+:)
2cnn_conv_1/kernel
:
2cnn_conv_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
6metrics
7layer_regularization_losses
8non_trainable_variables
trainable_variables
regularization_losses
	variables

9layers
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
:metrics
;layer_regularization_losses
<non_trainable_variables
trainable_variables
regularization_losses
	variables

=layers
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
+:)
2cnn_conv_2/kernel
:2cnn_conv_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
>metrics
?layer_regularization_losses
@non_trainable_variables
trainable_variables
regularization_losses
	variables

Alayers
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Bmetrics
Clayer_regularization_losses
Dnon_trainable_variables
trainable_variables
regularization_losses
 	variables

Elayers
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Fmetrics
Glayer_regularization_losses
Hnon_trainable_variables
"trainable_variables
#regularization_losses
$	variables

Ilayers
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
$:"
��2cnn_dense/kernel
:�2cnn_dense/bias
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
�
Jmetrics
Klayer_regularization_losses
Lnon_trainable_variables
(trainable_variables
)regularization_losses
*	variables

Mlayers
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
:	�
2dense/kernel
:
2
dense/bias
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
�
Nmetrics
Olayer_regularization_losses
Pnon_trainable_variables
.trainable_variables
/regularization_losses
0	variables

Qlayers
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
__inference__wrapped_model_5663�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_1���������
�2�
!__inference_cnn_layer_call_fn_342
!__inference_cnn_layer_call_fn_443
!__inference_cnn_layer_call_fn_355
!__inference_cnn_layer_call_fn_430�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
<__inference_cnn_layer_call_and_return_conditional_losses_122
<__inference_cnn_layer_call_and_return_conditional_losses_379
<__inference_cnn_layer_call_and_return_conditional_losses_205
<__inference_cnn_layer_call_and_return_conditional_losses_398�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_cnn_conv_1_layer_call_fn_151�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
C__inference_cnn_conv_1_layer_call_and_return_conditional_losses_144�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
+__inference_cnn_maxpool_1_layer_call_fn_132�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
F__inference_cnn_maxpool_1_layer_call_and_return_conditional_losses_127�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
'__inference_cnn_conv_2_layer_call_fn_52�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������

�2�
B__inference_cnn_conv_2_layer_call_and_return_conditional_losses_45�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������

�2�
*__inference_cnn_maxpool_2_layer_call_fn_11�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
D__inference_cnn_maxpool_2_layer_call_and_return_conditional_losses_6�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
)__inference_cnn_flatten_layer_call_fn_360�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_cnn_flatten_layer_call_and_return_conditional_losses_211�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_cnn_dense_layer_call_fn_255�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_cnn_dense_layer_call_and_return_conditional_losses_22�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
#__inference_dense_layer_call_fn_169�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
=__inference_dense_layer_call_and_return_conditional_losses_33�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
1B/
"__inference_signature_wrapper_5677input_1�
__inference__wrapped_model_5663y&',-8�5
.�+
)�&
input_1���������
� "3�0
.
output_1"�
output_1���������
�
C__inference_cnn_conv_1_layer_call_and_return_conditional_losses_144�I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������

� �
(__inference_cnn_conv_1_layer_call_fn_151�I�F
?�<
:�7
inputs+���������������������������
� "2�/+���������������������������
�
B__inference_cnn_conv_2_layer_call_and_return_conditional_losses_45�I�F
?�<
:�7
inputs+���������������������������

� "?�<
5�2
0+���������������������������
� �
'__inference_cnn_conv_2_layer_call_fn_52�I�F
?�<
:�7
inputs+���������������������������

� "2�/+����������������������������
A__inference_cnn_dense_layer_call_and_return_conditional_losses_22^&'0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� |
'__inference_cnn_dense_layer_call_fn_255Q&'0�-
&�#
!�
inputs����������
� "������������
D__inference_cnn_flatten_layer_call_and_return_conditional_losses_211a7�4
-�*
(�%
inputs���������
� "&�#
�
0����������
� �
)__inference_cnn_flatten_layer_call_fn_360T7�4
-�*
(�%
inputs���������
� "������������
<__inference_cnn_layer_call_and_return_conditional_losses_122n&',-;�8
1�.
(�%
inputs���������
p
� "%�"
�
0���������

� �
<__inference_cnn_layer_call_and_return_conditional_losses_205n&',-;�8
1�.
(�%
inputs���������
p 
� "%�"
�
0���������

� �
<__inference_cnn_layer_call_and_return_conditional_losses_379o&',-<�9
2�/
)�&
input_1���������
p 
� "%�"
�
0���������

� �
<__inference_cnn_layer_call_and_return_conditional_losses_398o&',-<�9
2�/
)�&
input_1���������
p
� "%�"
�
0���������

� �
!__inference_cnn_layer_call_fn_342b&',-<�9
2�/
)�&
input_1���������
p 
� "����������
�
!__inference_cnn_layer_call_fn_355a&',-;�8
1�.
(�%
inputs���������
p 
� "����������
�
!__inference_cnn_layer_call_fn_430b&',-<�9
2�/
)�&
input_1���������
p
� "����������
�
!__inference_cnn_layer_call_fn_443a&',-;�8
1�.
(�%
inputs���������
p
� "����������
�
F__inference_cnn_maxpool_1_layer_call_and_return_conditional_losses_127�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
+__inference_cnn_maxpool_1_layer_call_fn_132�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
D__inference_cnn_maxpool_2_layer_call_and_return_conditional_losses_6�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
*__inference_cnn_maxpool_2_layer_call_fn_11�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
=__inference_dense_layer_call_and_return_conditional_losses_33],-0�-
&�#
!�
inputs����������
� "%�"
�
0���������

� w
#__inference_dense_layer_call_fn_169P,-0�-
&�#
!�
inputs����������
� "����������
�
"__inference_signature_wrapper_5677�&',-C�@
� 
9�6
4
input_1)�&
input_1���������"3�0
.
output_1"�
output_1���������
