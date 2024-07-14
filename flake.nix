{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: {
    defaultPackage.x86_64-linux =
      # Notice the reference to nixpkgs here.
      with import nixpkgs { system = "x86_64-linux"; };
      let
        pkgs = import nixpkgs {
          inherit system;
          config = { allowUnfree = true; };
        };
      in 
        stdenvNoCC.mkDerivation {
          nativeBuildInputs = [
            pkgs.cmake
            pkgs.clang-tools
            pkgs.python312
            pkgs.python312Packages.cython
            pkgs.doxygen
            pkgs.swig
            pkgs.gnumake
            pkgs.cudatoolkit
            #pkgs.cudaPackages.cuda_opencl
            pkgs.opencl-clhpp
            pkgs.ocl-icd
            pkgs.clang
            pkgs.lldb
          ];
          #buildInputs = [];
          name = "hello";
          src = self;
        };
  };
}
