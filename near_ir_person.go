package near_ir_person

import (
	"context"
	"runtime"

	ort "github.com/yalue/onnxruntime_go"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/ml"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/mlmodel"
	"gorgonia.org/tensor"
)

const modelPath = "./ir_mobilenet.onnx"

var Model = resource.ModelNamespace("bhaney").WithFamily("mlmodel").WithModel("near_ir_person")
var blank []uint8

func init() {
	resource.RegisterService(mlmodel.API, Model, resource.Registration[mlmodel.Service, *Config]{
		Constructor: func(ctx context.Context, deps resource.Dependencies, conf resource.Config, logger logging.Logger) (mlmodel.Service, error) {
			newConf, err := resource.NativeConfig[*Config](conf)
			if err != nil {
				return nil, err
			}

			nirp, err := initModel(conf.ResourceName(), logger)
			if err != nil {
				return nil, err
			}

			return nirp, nil
		},
	})
}

type Config struct {
	resource.TriviallyValidateConfig
}

type modelSession struct {
	Session *ort.AdvancedSession
	Input   *ort.Tensor[uint8]
	Output  []*ort.Tensor[float32]
}

type nearIRPerson struct {
	resource.AlwaysRebuild
	name     resource.Name
	logger   logging.Logger
	session  modelSession
	metadata mlmodel.MLMetadata
}

func initModel(name resource.Name, logger logging.Logger) (*nearIRPerson, error) {
	nirp := &nearIRPerson{name: name, logger: logger}
	libPath, err := getSharedLibPath()
	if err != nil {
		return nil, err
	}
	ort.SetSharedLibraryPath(libPath)
	err := ort.InitializeEnvironment()
	if err != nil {
		return nil, err
	}
	// create the metadata
	nirp.metadata = createMetadata()
	// declare the input Tensor for the near IR person model
	// the image
	inputShape := ort.NewShape(1, 300, 300, 3)
	inputTensor, err := ort.NewTensor(inputShape, blank)
	if err != nil {
		return nil, err
	}

	// declare the output Tensors for the near IR person model
	outputTensors := make([]*ort.Tensor[float32], 8)

	// detection_anchor_indices
	outputShape0 := ort.NewShape(1, 100)
	outputTensor0, err := ort.NewEmptyTensor[float32](outputShape0)
	if err != nil {
		return nil, err
	}
	outputTensors[0] = outputTensor0
	// detection_boxes
	outputShape1 := ort.NewShape(1, 100, 4)
	outputTensor1, err := ort.NewEmptyTensor[float32](outputShape1)
	if err != nil {
		return nil, err
	}
	outputTensors[1] = outputTensor1
	// detection_classes
	outputShape2 := ort.NewShape(1, 100)
	outputTensor2, err := ort.NewEmptyTensor[float32](outputShape2)
	if err != nil {
		return nil, err
	}
	outputTensors[2] = outputTensor2
	// detection_multiclass_scores
	outputShape3 := ort.NewShape(1, 100, 2)
	outputTensor3, err := ort.NewEmptyTensor[float32](outputShape3)
	if err != nil {
		return nil, err
	}
	outputTensors[3] = outputTensor3
	// detection_scores
	outputShape4 := ort.NewShape(1, 100)
	outputTensor4, err := ort.NewEmptyTensor[float32](outputShape4)
	if err != nil {
		return nil, err
	}
	outputTensors[4] = outputTensor4
	//num_detections
	outputShape5 := ort.NewShape(1)
	outputTensor5, err := ort.NewEmptyTensor[float32](outputShape5)
	if err != nil {
		return nil, err
	}
	outputTensors[5] = outputTensor5
	// raw_detection_boxes
	outputShape6 := ort.NewShape(1, 1917, 4)
	outputTensor6, err := ort.NewEmptyTensor[float32](outputShape6)
	if err != nil {
		return nil, err
	}
	outputTensors[6] = outputTensor6
	// raw_detection_scores
	outputShape7 := ort.NewShape(1, 1917, 2)
	outputTensor7, err := ort.NewEmptyTensor[float32](outputShape7)
	if err != nil {
		return nil, err
	}
	outputTensors[7] = outputTensor7

	options, e := ort.NewSessionOptions()
	if e != nil {
		return nil, err
	}

	arbitraryOutput := make([]ort.ArbitraryTensor, len(outputTensors))
	for i, tensor := range outputTensors {
		arbitraryOutput[i] = tensor
	}
	session, err := ort.NewAdvancedSession(modelPath,
		[]string{"input_tensor"},
		[]string{
			"detection_anchor_indices",
			"detection_boxes",
			"detection_classes",
			"detection_multiclass_scores",
			"detection_scores",
			"num_detections",
			"raw_detection_boxes",
			"raw_detection_scores"},
		[]ort.ArbitraryTensor{inputTensor}, arbitraryOutput, options)

	modelSes := modelSession{
		Session: session,
		Input:   inputTensor,
		Output:  outputTensors,
	}
	nirp.session = modelSes

	return nirp, nil
}

func (nirp *nearIRPerson) Name() resource.Name {
	return nirp.name
}

func (nirp *nearIRPerson) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	return nil, resource.ErrDoUnimplemented
}

func (nirp *nearIRPerson) Infer(ctx context.Context, tensors ml.Tensors) (ml.Tensors, error) {
	input, err := processInput(tensors)
	if err != nil {
		return nil, err
	}
	inTensor := nirp.session.Input.GetData()
	copy(inTensor, input)
	err := nirp.session.Session.Run()
	if err != nil {
		return nil, err
	}
	outputData := make([][]float32, 0, 8)
	for _, out := range nirp.session.Output {
		if outData, ok := out.GetData().([]float32); ok {
			outputData = append(outputData, outData)
		} else {
			return nil, errors.New("could not convert outputs tensor into []float32")
		}
	}
	return processOutputs(outputData)
}

func processInput(inputs ml.Tensors) ([]uint8, error) {
	var imageTensor *tensor.Dense
	// if length of tensors is 1, just grab the first tensor
	// if more than 1 grab the one called input tensor, or image
	if len(tensors) == 1 {
		for _, t := range tensors {
			imageTensor = t
			break
		}
	} else {
		for name, t := range tensors {
			if name == "image" || name == "input_tensor" {
				imageTensor = t
				break
			}
		}
	}
	if imageTensor == nil {
		return nil, errors.New("no valid input tensor called 'image' or 'input_tensor' found")
	}
	if uint8Data, ok := imageTensor.Data().([]uint8); ok {
		return uint8Data, nil
	}
	return nil, errors.Errorf("input tensor must be of tensor type UIn8, got %v", imageTensor.Dtype())
}

func processOutput(outputs [][]float32) (ml.Tensors, error) {
	// there are 8 output tensors. Turn them into tensors with the right backing
	outMap = ml.Tensors{}
	outMap["detection_anchor_indices"] = tensor.New(
		tensor.WithShape(1, 100),
		tensors.WithBacking(outputs[0]),
	)
	outMap["location"] = tensor.New(
		tensor.WithShape(1, 100, 4),
		tensors.WithBacking(outputs[1]),
	)
	outMap["category"] = tensor.New(
		tensor.WithShape(1, 100),
		tensors.WithBacking(outputs[2]),
	)
	outMap["detection_multiclass_scores"] = tensor.New(
		tensor.WithShape(1, 100, 2),
		tensors.WithBacking(outputs[3]),
	)
	outMap["score"] = tensor.New(
		tensor.WithShape(1, 100),
		tensors.WithBacking(outputs[4]),
	)
	outMap["num_detections"] = tensor.New(
		tensor.WithShape(1),
		tensors.WithBacking(outputs[5]),
	)
	outMap["raw_detection_boxes"] = tensor.New(
		tensor.WithShape(1, 1917, 2),
		tensors.WithBacking(outputs[6]),
	)
	outMap["raw_detection_scores"] = tensor.New(
		tensor.WithShape(1, 1917, 2),
		tensors.WithBacking(outputs[7]),
	)
	return outMap, nil
}

func (nirp *nearIRPerson) Metadata(ctx context.Context) (mlmodel.MLMetadata, error) {
	return nirp.metadata, nil
}

func (nirp *nearIRPerson) Close(ctx context.Context) error {
	// destroy session
	err := nirp.session.Session.Destroy()
	if err != nil {
		return err
	}
	// destroy tensors
	err := nirp.session.Output.Destroy()
	if err != nil {
		return err
	}
	err := nirp.session.Input.Destroy()
	if err != nil {
		return err
	}
	// destroy environment
	err := ort.DestroyEnvironment()
	if err != nil {
		return err
	}
	return nil
}

func getSharedLibPath() (string, error) {
	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			return "./third_party/onnxruntime.dll", nil
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return "./third_party/onnxruntime_arm64.dylib", nil
		}
	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			return "./third_party/onnxruntime_arm64.so", nil
		}
		return "./third_party/onnxruntime.so", nil
	}
	return "", errors.Errorf("Unable to find a version of the onnxruntime library supporting %s %s", runtime.GOOS, runtime.GOARCH)
}

func createMetadata() mlmodel.MLMetadata {
	md := mlmodel.MLMetadata{}
	md.ModelName = "near_ir_person_ssd_mobilenetv2"
	md.ModelType = "object_detector"
	md.ModelDescription = "an SSD MobileNetV2 model in ONNX format used to detect people from a Near IR camera. Takes RGB input"
	// inputs
	inputs := []mlmodel.TensorInfo{}
	imageIn := mlmodel.TensorInfo{
		Name:     "input_tensor",
		DataType: "uint8",
		Shape:    []int{1, 300, 300, 3},
	}
	inputs = append(inputs, imageIn)
	md.Inputs = inputs
	// outputs
	outputs := []mlmodel.TensorInfo{}
	out0 := mlmodel.TensorInfo{
		Name:     "detection_anchor_indices",
		DataType: "float32",
		Shape:    []int{1, 100},
	}
	outputs = append(outputs, out0)
	out1 := mlmodel.TensorInfo{
		Name:     "location",
		DataType: "float32",
		Shape:    []int{1, 100, 4},
	}
	outputs = append(outputs, out1)
	out2 := mlmodel.TensorInfo{
		Name:     "category",
		DataType: "float32",
		Shape:    []int{1, 100},
	}
	outputs = append(outputs, out2)
	out3 := mlmodel.TensorInfo{
		Name:     "detection_multiclass_scores",
		DataType: "float32",
		Shape:    []int{1, 100, 2},
	}
	outputs = append(outputs, out3)
	out4 := mlmodel.TensorInfo{
		Name:     "score",
		DataType: "float32",
		Shape:    []int{1, 100},
	}
	outputs = append(outputs, out4)
	out5 := mlmodel.TensorInfo{
		Name:     "num_detections",
		DataType: "float32",
		Shape:    []int{1},
	}
	outputs = append(outputs, out5)
	out6 := mlmodel.TensorInfo{
		Name:     "raw_detection_boxes",
		DataType: "float32",
		Shape:    []int{1, 1917, 2},
	}
	outputs = append(outputs, out6)
	out7 := mlmodel.TensorInfo{
		Name:     "raw_detection_scores",
		DataType: "float32",
		Shape:    []int{1, 1917, 2},
	}
	outputs = append(outputs, out7)
	md.Outputs = outputs
	return md
}
