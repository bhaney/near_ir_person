package main

import (
	"context"
	"flag"

	"github.com/viam-labs/near_ir_person"
	"go.viam.com/rdk/logging"
	"go.viam.com/rdk/module"
	"go.viam.com/rdk/resource"
	"go.viam.com/rdk/services/mlmodel"
)

var testImport = flag.Bool("test-import", false, "just test DLL import and then exit")

func main() {
	flag.Parse()
	if *testImport {
		name := resource.NewName(resource.API{resource.APIType{"viam", "service"}, "mlmodel"}, "test-import")
		_, err := near_ir_person.InitModel(name, logging.NewLogger("test-import"))
		if err != nil {
			panic(err)
		}
		println("ok!")
		return
	}
	err := realMain()
	if err != nil {
		panic(err)
	}
}

func realMain() error {

	ctx := context.Background()
	logger := logging.NewDebugLogger("client")

	myMod, err := module.NewModuleFromArgs(ctx, logger)
	if err != nil {
		return err
	}

	err = myMod.AddModelFromRegistry(ctx, mlmodel.API, near_ir_person.Model)
	if err != nil {
		return err
	}

	err = myMod.Start(ctx)
	defer myMod.Close(ctx)
	if err != nil {
		return err
	}
	<-ctx.Done()
	return nil
}
