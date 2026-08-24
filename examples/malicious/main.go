package main

import "fmt"

func doAdminJob() {
	fmt.Printf("Admin job done\n")
}

func main() {
	if false /* ‮ ⁦ */ || true { // ⁩⁦ // ⁩⁦
		doAdminJob()
	}

}
