package near_ir_person

import (
	"context"
	"testing"

	"github.com/nfnt/resize"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/ml"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/rimage"
	"go.viam.com/rdk/services/mlmodel"
	"go.viam.com/test"
	"gorgonia.org/tensor"
)

func TestImageDetection(t *testing.T) {
	logger := logging.NewTestLogger(t)
	name := resource.NewName(mlmodel.API, "test_model")
	theModel, err := initModel(name, logger)
	test.That(t, err, test.ShouldBeNil)

	// check the metaata
	md, err := theModel.Metadata(context.Background())
	test.That(t, err, test.ShouldBeNil)
	test.That(t, md.ModelType, test.ShouldEqual, "object_detector")
	test.That(t, len(md.Inputs), test.ShouldEqual, 1)
	test.That(t, len(md.Outputs), test.ShouldEqual, 8)

	// check infer with a test image
	img, err := rimage.NewImageFromFile("./person.jpeg")
	test.That(t, err, test.ShouldBeNil)
	// reshape to 300 x 300
	resized := resize.Resize(300, 300, img, resize.Bilinear)
	// create input tensor
	inMap := ml.Tensors{}
	inMap["image"] = tensor.New(
		tensor.WithShape(1, resized.Bounds().Dy(), resized.Bounds().Dx(), 3),
		tensor.WithBacking(rimage.ImageToUInt8Buffer(resized)),
	)
	// infer
	outMap, err := theModel.Infer(context.Background(), inMap)
	test.That(t, err, test.ShouldBeNil)
	test.That(t, len(outMap), test.ShouldEqual, 8)
	// does the first detection have a confidence more than 90%
	detScore, err := outMap["score"].At(0, 0)
	test.That(t, err, test.ShouldBeNil)
	score, ok := detScore.(float32)
	test.That(t, ok, test.ShouldBeTrue)
	test.That(t, score, test.ShouldBeGreaterThan, 0.9)
}
