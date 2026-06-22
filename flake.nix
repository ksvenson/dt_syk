{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-24.05";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = with pkgs; [
          openmpi
          gcc
          gnumake

          # Transparent wrapper to fix SCALAPACK's legacy policy
          (pkgs.writeShellScriptBin "cmake" ''
            exec ${pkgs.cmake}/bin/cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 "$@"
          '')

          cacert
          libxcrypt-legacy
        ];
        shellHook = ''
          # Needed to download packages from the internet:
          # Point tools to the correct SSL certificate bundle
          export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
          export GIT_SSL_CAINFO=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
          echo "Loading C packages."

          # Needed for python 3.9.7:
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ 
            pkgs.libxcrypt-legacy
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ]}:$LD_LIBRARY_PATH"
        '';
      };
    };
}
