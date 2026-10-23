# Flake: Granite 4.2 8B on one RTX 4070 Ti Super (sm_89) scaffold.
#
# Outputs per system (see the `outputs` attribute below):
#   packages.<system>.engine   host-side C++ library (C ABI, safetensors
#                              metadata, JSON parser, config validation);
#                              built host-only - CUDA objects target sm_89
#                              and are enabled on the GPU host, not here
#   packages.<system>.api      Python stdlib API frontend (Phase 0:
#                              validates requests, no inference)
#   packages.<system>.default  = api
#   devShell.<system>          cmake + ninja + clang + git + uv + python3
#                              (regex, pytest), GRANITE4_LIB preset
#   checks.<system>            engine (runs ctest) and api (runs pytest)
#   nixosModules.granite4Rtx   NixOS systemd service for the api process
#
# The 18 GB local checkpoint under .gt/ and the build trees never enter any
# derivation: the source filter below keeps only the files the derivations
# need, so the source hash does not depend on the local checkpoint or on
# whether the git submodule under reference/ is initialised.
{
  description = "Granite 4.2 8B on one RTX 4070 Ti Super (sm_89) - C++/CUDA engine + stdlib Python API scaffold";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs =
    { self, nixpkgs }:
    let
      # The flake's own working tree. `./.` is resolved relative to this
      # flake's source; using it (rather than self.source) keeps the flake
      # buildable from a plain checkout where self.source is undefined.
      source = ./.;

      systems = [ "aarch64-linux" "x86_64-linux" "aarch64-darwin" "x86_64-darwin" ];

      # Map each system to its nixpkgs and pass it to f as a single
      # positional argument (no destructuring - this Nix build is strict
      # about function-argument attrs).
      forAll =
        f:
        nixpkgs.lib.genAttrs systems (system: f (import nixpkgs { inherit system; }));

      # Keep only the files the derivations need. filterSource visits every
      # entry with an absolute path, so exclusions are absolute prefixes.
      # Drops:
      #   - the 18 GB local checkpoint (.gt/), build trees, and caches
      #   - the git submodule under reference/ (its content is not part of
      #     this project - see the "layout conventions only" note in the
      #     README); dropping it keeps the source hash stable whether or not
      #     the submodule is initialised
      submodulePrefix = toString source + "/reference/qwen3-27b-5090";

      # Returns true for entries to keep. The checkpoint, build trees, caches
      # and the git submodule are directories; excluding the directory prunes
      # its entire subtree, so the 18 GB .gt/ checkpoint and the submodule
      # never enter the source (and the hash does not depend on whether the
      # submodule is initialised).
      keepSource =
        path:
        type:
        let
          base = nixpkgs.lib.baseNameOf path;
          inSubmodule =
            nixpkgs.lib.hasPrefix submodulePrefix path
            || nixpkgs.lib.hasPrefix (submodulePrefix + "/") path;
          excludedName =
            nixpkgs.lib.any (n: base == n) [ ".git" ".gt" ".venv" "build" "build-host" ".pytest_cache" "__pycache__" ];
        in
        !(type == "directory" && (inSubmodule || excludedName));

      repo = builtins.filterSource keepSource source;

      libext = pkgs: if pkgs.stdenv.hostPlatform.isDarwin then "dylib" else "so";

      mkEngine =
        pkgs:
        pkgs.stdenv.mkDerivation {
          pname = "granite4-engine";
          version = "0.0.1";
          src = repo;
          # Run the CTest suite (config_validation, safetensors_metadata,
          # json_parser) inside the derivation so checks.* is meaningful.
          doCheck = true;
          nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.clang ];
          configurePhase = ''
            # CMAKE_CXX_SCAN_FOR_MODULES=OFF: CMake 4 runs clang-scan-deps as
            # a separate pre-compile pass; that subprocess does not receive the
            # sysroot the real clang invocation does, so it fails to find the
            # C++ standard headers. Disabling the scan keeps the real build.
            cmake -B build -S . -G Ninja \
              -DGRANITE_ENABLE_CUDA=OFF \
              -DCMAKE_CXX_SCAN_FOR_MODULES=OFF
          '';
          buildPhase = ''
            # Override stdenv's default buildPhase (which would run a bare
            # `make`); the Ninja build is driven through CMake instead.
            cmake --build build
          '';
          # Runs config_validation, safetensors_metadata and json_parser.
          checkPhase = "ctest --test-dir build --output-on-failure";
          installPhase = ''
            mkdir -p $out/lib $out/include/granite4
            install -m 0755 build/engine/libgranite4.${libext pkgs} $out/lib/
            install -m 0644 engine/src/*.h $out/include/granite4/
          '';
          meta.description = "Granite 4.2 8B host-side engine library (C ABI; CUDA off, sm_89 on the GPU host)";
        };

      # Pure-stdlib frontend. format = "other" skips the setuptools path; we
      # install the frontend/ and reference/ packages into site-packages and
      # expose a thin granite4-api launcher. The launcher must run in a clean
      # environment (systemd): it execs the Nix python interpreter that
      # carries regex, and adds $out/lib to PYTHONPATH explicitly.
      mkApi =
        pkgs:
        let
          python = pkgs.python3;
          interpreter = python.withPackages (ps: [ ps.regex ]);
        in
        python.pkgs.buildPythonApplication {
          pname = "granite4-api";
          version = "0.0.1";
          src = repo;
          format = "other";
          buildInputs = [ python.pkgs.regex ];
          installPhase = ''
            mkdir -p $out/lib/$sitePackages
            cp -r frontend $out/lib/$sitePackages/
            cp -r reference $out/lib/$sitePackages/
            mkdir -p $out/bin
            # Launcher is a Python script with a shebang for the Nix
            # interpreter (the withPackages env does not bundle /bin/sh). It
            # prepends $out/lib to sys.path so `frontend` / `reference` resolve
            # in a clean environment such as a systemd service.
            printf '%s\n' \
              '#!${interpreter}/bin/python3' \
              'import os, sys' \
              '_lib = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")' \
              'if _lib not in sys.path:' \
              '    sys.path.insert(0, _lib)' \
              'from frontend.api import main' \
              'main()' \
              > $out/bin/granite4-api
            chmod 0755 $out/bin/granite4-api
          '';
          meta.description = "Granite 4.2 8B API frontend (Phase 0 scaffold: validates requests, no inference)";
        };

      # checks.api additionally runs the Python test suite. It uses an
      # interpreter carrying both regex and pytest so that frontend.tokenizer's
      # `import regex` resolves and `python -m pytest` needs no separate binary.
      mkApiChecked =
        pkgs:
        let
          python = pkgs.python3;
          checker = python.withPackages (ps: [ ps.regex ps.pytest ]);
          app = mkApi pkgs;
        in
        app.overrideAttrs (final: prev: {
          doCheck = true;
          checkPhase = ''
            PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" ${checker}/bin/python -m pytest -q tests/
          '';
        });
    in
    {
      packages =
        forAll (
          pkgs:
          {
            engine = mkEngine pkgs;
            api = mkApi pkgs;
            default = mkApi pkgs;
          }
        );

      devShell =
        forAll (
          pkgs:
          let
            python = pkgs.python3;
            ext = libext pkgs;
          in
          pkgs.mkShell {
            name = "granite4-rtx-dev";
            packages = [
              pkgs.cmake
              pkgs.ninja
              pkgs.clang
              pkgs.git
              pkgs.uv
              python
              python.pkgs.regex
              python.pkgs.pytest
            ];
            shellHook = ''
              # /metrics probes the optional engine library when present.
              export GRANITE4_LIB="$PWD/build/engine/libgranite4.${ext}"
              export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
            '';
          }
        );

      checks =
        forAll (
          pkgs:
          {
            engine = mkEngine pkgs;
            api = mkApiChecked pkgs;
          }
        );

      # NixOS module for the deployment target (docs/deployment.md Step 5).
      # The remote flake imports this repo as a flake input and enables
      # services.granite4-rtx, passing the api package from that input.
      # The module never references this flake's self.
      nixosModules.granite4Rtx =
        { config, lib, ... }:
        let
          cfg = config.services.granite4-rtx;
        in
        {
          options.services."granite4-rtx" = {
            enable = lib.mkEnableOption "the Granite 4.2 8B API scaffold service";
            package = lib.mkOption {
              description = "The granite4-api derivation (flake packages.<system>.api)";
              type = lib.types.package;
            };
            listenAddress = lib.mkOption {
              type = lib.types.str;
              default = "127.0.0.1";
            };
            listenPort = lib.mkOption {
              type = lib.types.port;
              default = 8080;
            };
          };

          config =
            lib.mkIf cfg.enable {
              users = {
                groups."granite4-rtx" = { };
                users."granite4-rtx" = {
                  isSystemUser = true;
                  group = "granite4-rtx";
                  description = "Granite 4.2 8B API scaffold";
                };
              };
              systemd.services.granite4-rtx = {
                description = "Granite 4.2 8B API scaffold (Phase 0: validates requests, no inference)";
                after = [ "network.target" ];
                wantedBy = [ "multi-user.target" ];
                serviceConfig = {
                  User = "granite4-rtx";
                  Group = "granite4-rtx";
                  ExecStart = "${cfg.package}/bin/granite4-api --host ${cfg.listenAddress} --port ${toString cfg.listenPort}";
                  Restart = "on-failure";
                  RestartSec = 5;
                  # The Phase 0 process is read-only against the system: it
                  # only opens a listening socket.
                  NoNewPrivileges = true;
                  ProtectSystem = "strict";
                  ProtectHome = true;
                  PrivateTmp = true;
                  PrivateDevices = true;
                  RestrictAddressFamilies = [ "af_INET" "af_INET6" ];
                  CapabilityBoundingSet = "";
                };
              };
            };
        };
    };

}
