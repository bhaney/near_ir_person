package near_ir_person

import (
	"context"

	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/ml"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/mlmodel"
)

var Model = resource.ModelNamespace("bhaney").WithFamily("mlmodel").WithModel("near_ir_person")

func init() {
	resource.RegisterService(mlmodel.API, Model, resource.Registration[mlmodel.Service, *Config]{
		Constructor: func(ctx context.Context, deps resource.Dependencies, conf resource.Config, logger logging.Logger) (mlmodel.Service, error) {
			newConf, err := resource.NativeConfig[*Config](conf)
			if err != nil {
				return nil, err
			}

			nirp := &nearIRPerson{name: conf.ResourceName(), conf: newConf, logger: logger}

			return nirp, nil
		},
	})
}

type Config struct {
}

func (cfg *Config) Validate(path string) ([]string, error) {
}

type nearIRPerson struct {
	resource.AlwaysRebuild
	resource.TriviallyCloseable

	name   resource.Name
	conf   *Config
	logger logging.Logger
}

func (nirp *nearIRPerson) Name() resource.Name {
	return nirp.name
}

func (nirp *nearIRPerson) DoCommand(ctx context.Context, cmd map[string]interface{}) (map[string]interface{}, error) {
	return nil, resource.ErrDoUnimplemented
}

func (nirp *nearIRPerson) Infer(ctx context.Context, tensors ml.Tensors) (ml.Tensors, error) {
	return nil, nil
}

func (nirp *nearIRPerson) Metadata(ctx context.Context) (mlmodel.MLMetadata, error) {
	return mlmodel.MLMetadata{}, nil
}

func (nirp *nearIRPerson) Close(ctx context.Context) error {
	return nil
}
