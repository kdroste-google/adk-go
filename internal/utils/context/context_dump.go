// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package utils

import (
	"context"
	"log"
	"reflect"
	"strings"
)

func DumpToLog(ctx context.Context, msg string) {
	n := 1000
	dumpCtxRec(0, 100, &n, ctx)

}

func formatVal(v reflect.Value, t reflect.Type) string {
	// if v.IsNil() {
	// 	return "<nil>"
	// }
	// log.Printf(">> t: %v v: %v, elem: %v", t.String(), v.String(), v.Elem())

	// log.Printf("addr %v, complex %v, float %v, int %v, interface %v, uint %v, set %v, kind %v, type %v",
	// 	v.CanAddr(), v.CanComplex(), v.CanFloat(), v.CanInt(),
	// 	v.CanInterface(), v.CanUint(), v.CanSet(), v.Kind(), v.Type())

	// if v.Type().String() == "interface {}" {
	// 	return fmt.Sprintf("%+v", v.Interface())
	// }
	// log.Printf("v:%v kind: %v canInterface: %v", v, v.Kind(), v.CanInterface())
	return v.Elem().String()
}

func dumpCtxRec(d int, maxD int, n *int, o any) {
	ind := strings.Repeat("  ", d)
	if d >= maxD {
		log.Printf("%sMax depth reached", ind)
		return
	}
	if *n <= 0 {
		log.Printf("%sReached max number of objects", ind)
		return
	}
	*n--
	ot := reflect.TypeOf(o)

	if ot.Kind() == reflect.Pointer {
		log.Printf("%sPointer to %s", ind, ot.String())
		o = reflect.Indirect(reflect.ValueOf(o)).Interface()
		dumpCtxRec(d+1, maxD, n, o)
		return
	}
	if ot.Kind() == reflect.Struct {
		ov := reflect.ValueOf(o)
		log.Printf("%sStruct %s", ind, ot.String())

		todos := []any{}

		switch ot.String() {
		case "context.valueCtx":
			var key, val reflect.Value
			var keyT, valT reflect.Type
			for i := 0; i < ot.NumField(); i++ {
				f := ot.Field(i)

				switch f.Name {
				case "Context":
					todos = append(todos, ov.Field(i).Interface())
				case "key":
					key = ov.Field(i)
					keyT = f.Type
				case "val":
					val = ov.Field(i)
					valT = f.Type
				default:
					log.Printf("%s  Unknown Field %s: %s", ind, f.Name, f.Type.String())
				}
			}
			log.Printf("%skey: %v", ind, formatVal(key, keyT))
			log.Printf("%sval: %v", ind, formatVal(val, valT))
			// if !key.IsNil() {
			// 	log.Printf("%skey: %v (type %T kind: %v)", ind, key, key, key.Kind())
			// }
			// if !val.IsNil() {
			// 	log.Printf("%s val: %+v (type %T kind: %v)", ind, val, val, val.Kind())
			// }

		case "agent.invocationContext":
			for i := 0; i < ot.NumField(); i++ {
				f := ot.Field(i)

				switch f.Name {
				case "Context":
					todos = append(todos, ov.Field(i).Interface())
				case "agent":
				case "artifacts":
				case "memory":
				case "session":
				case "invocationID":
				case "branch":
				case "userContent":
				case "runConfig":
				case "endInvocation":
				default:
					log.Printf("%s  Unknown Field %s: %s", ind, f.Name, f.Type.String())
				}
			}
		case "context.InvocationContext":
			for i := 0; i < ot.NumField(); i++ {
				f := ot.Field(i)

				switch f.Name {
				case "Context":
					todos = append(todos, ov.Field(i).Interface())
				case "params":
				default:
					log.Printf("%s  Unknown Field %s: %s", ind, f.Name, f.Type.String())
				}
			}
		case "signal.signalCtx":
			for i := 0; i < ot.NumField(); i++ {
				f := ot.Field(i)

				switch f.Name {
				case "Context":
					todos = append(todos, ov.Field(i).Interface())
				case "cancel":
				case "signals":
				case "ch":
				default:
					log.Printf("%s  Unknown Field %s: %s", ind, f.Name, f.Type.String())
				}
			}
		case "context.cancelCtx":
			for i := 0; i < ot.NumField(); i++ {
				f := ot.Field(i)

				switch f.Name {
				case "Context":
					todos = append(todos, ov.Field(i).Interface())
				case "mu":
				case "done":
				case "children":
					children := ov.Field(i).MapKeys()
					log.Printf("%schildren: %v nElems: %v", ind, children, len(children))
					for _, child := range children {
						todos = append(todos, child)
					}

				case "err":
				case "cause":

				default:
					log.Printf("%s  Unknown Field %s: %s", ind, f.Name, f.Type.String())
				}
			}
		case "context.backgroundCtx":
			for i := 0; i < ot.NumField(); i++ {
				f := ot.Field(i)

				switch f.Name {
				case "emptyCtx":
				default:
					log.Printf("%s  Unknown Field %s: %s", ind, f.Name, f.Type.String())
				}
			}

		default:
			log.Printf("%sUnknown struct %s", ind, ot.String())
		}

		for _, todo := range todos {
			dumpCtxRec(d+1, maxD, n, todo)
		}

		return
	}

	log.Printf("%sUnknown", ind)

}

// for {
// 	depth--
// 	if depth == 0 {
// 		break
// 	}
// 	t := reflect.TypeOf(ctx)
// 	v := reflect.ValueOf(ctx)
// 	log.Printf("Type: %v kind %v", t.String(), t.Kind())

// 	switch t.Kind() {
// 	case reflect.Pointer:
// 		ctx = reflect.Indirect(reflect.ValueOf(ctx)).Interface()
// 		continue
// 	case reflect.Struct:
// 		for i := 0; i < t.NumField(); i++ {
// 			f := t.Field(i)
// 			log.Printf("  %s: %s", f.Name, f.Type.String())
// 			fv := v.Field(i)
// 		}
// 		return

// 	default:
// 		return
// 	}

// ctxs = append(ctxs, ctxEntry{t, t.String()})
// s := t.String()
// switch s {
// case "*context.valueCtx":
// 	for i := 0; i < t.NumField(); i++ {
// 		f := t.Field(i)
// 		log.Printf("  %s: %s", f.Name, f.Type.String())
// 		// ctxs = append(ctxs, ctxEntry{f.Type, f.Type.String()})
// 	}

// // case "context.Context":
// default:
// 	log.Printf("Default: %v", s)

// }
// 	break
// }

// }
