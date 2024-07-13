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
      stdenv.mkDerivation {
        nativeBuildInputs = [ pkgs.cmake ];
        buildInputs = [
          pkgs.python312
          pkgs.python312Packages.cython
          pkgs.doxygen
          pkgs.swig
          pkgs.gnumake
          pkgs.cudatoolkit
          pkgs.clang
          pkgs.clang-tools
          pkgs.lldb
        ];
        name = "hello";
        src = self;
        # configurePhase = "cmake -B .";
        # buildPhase = "cmake --build . --parallel $NIX_BUILD_CORES";
        # installPhase = "mkdir -p $out/bin; install -t $out/bin hello";
      };
  };
}
