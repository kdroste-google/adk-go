package main

import (
	"fmt"
	"reflect"
)

type Input struct {
	AnInt   int
	AString string
	ArrInts []int
}

type Output struct {
	ASecondInt int
}

func DoIt(in Input) (Output, error) {
	return Output{ASecondInt: in.AnInt * 2}, nil
}

func main() {
	refl(DoIt)
}

func refl(f func(in Input) (Output, error)) error {
	t := reflect.TypeOf(f)
	if t.Kind() != reflect.Func {
		return fmt.Errorf("t.Kind() is not a function: %v", t.Kind())
	}
	for i := 0; i < t.NumIn(); i++ {
		inParam := t.In(i)
		fmt.Printf("inParam[%d]: %+v\n", i, inParam)
		inParam.T

	}

	fmt.Printf("%+v", t)
	return nil
}
