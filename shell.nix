let

  nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-23.11";

  pkgs = import nixpkgs { config = {}; overlays = []; };

in


pkgs.mkShellNoCC {

  packages = with pkgs; [
    python312
    python312Packages.cython
    cmake
    doxygen
    swig
    gnumake
    cudatoolkit
    cudaPackages.cudnn
  ];

}