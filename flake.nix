# Flake: Granite 4.2 8B on one RTX 4070 Ti Super (sm_89) scaffold.
#
# Outputs per system (see the `outputs` attribute below):
#   packages.<system>.engine     host-side C++ library (C ABI, safetensors
#                                metadata, JSON parser, config validation);
#                                built host-only - CUDA objects target sm_89
#                                and are enabled on the GPU host, not here
#   packages.<system>.api        Python stdlib API frontend + parity gate
#                                (granite4-api, granite4-parity-gate)
#   packages.<system>.default    = api
#   packages.<system>.engine-cuda  CUDA engine (x86_64-linux only): compiles
#                                the sm_89 kernels, device test, micro-
#                                benchmarks (bandwidth + GEMV), and the
#                                inference CLI (--gpu 1 runs the device
#                                forward); doCheck runs all ctest suites
#                                except the GPU-only kernels_device, which
#                                needs the host driver's libcuda + a live
#                                device and is re-run on gremlin-1
#   devShell.<system>            cmake + ninja + clang + git + uv + python3
#                                (regex, pytest), GRANITE4_LIB preset
#   checks.<system>              engine (runs ctest) and api (runs pytest)
#   nixosModules.granite4Rtx     NixOS service: the api process, plus (with a
#                                checkpoint) a GPU parity gate on start and a
#                                persistent GPU engine behind it
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
            mkdir -p $out/lib $out/bin $out/include/granite4
            install -m 0755 build/engine/libgranite4.${libext pkgs} $out/lib/
            # granite4-checkpoint is linked against the static library
            # (granite4_static), so it carries no rpath into the build tree
            # and runs from the installed profile as-is.
            install -m 0755 build/engine/granite4-checkpoint $out/bin/
            install -m 0644 engine/src/*.h $out/include/granite4/
          '';
          meta.description = "Granite 4.2 8B host-side engine library (C ABI; CUDA off, sm_89 on the GPU host)";
        };

      # The CUDA-enabled engine (sm_89). Linux-only: it needs the CUDA toolkit
      # to compile the device kernels. doCheck compiles the whole sm_89 build
      # (kernels, device test, micro-benchmark) and runs every ctest suite
      # except kernels_device - that suite needs the GPU host driver's
      # libcuda plus a live sm_89 device, which a build sandbox cannot reach;
      # it is re-verified on the GPU box with one command:
      #   granite4-kernel-test      (device correctness)
      #   granite4-kernel-bench     (measured bandwidth / roofline baseline;
      #                              JSON to stdout - the measured numbers are
      #                              recorded outside the repo per the P4 rule)
      mkEngineCuda =
        pkgs:
        let
          # allowUnfree only for this derivation's stdenv: the CUDA components
          # are unfree-licensed; nothing else in the repo needs it. Same
          # nixpkgs input, so shared inputs (cmake/ninja/clang, libc) are the
          # identical store paths as in the host-only engine.
          p = import nixpkgs {
            inherit (pkgs.stdenv.hostPlatform) system;
            config.allowUnfree = true;
          };
        in
        p.stdenv.mkDerivation {
          pname = "granite4-engine-cuda";
          version = "0.0.1";
          src = repo;
          doCheck = true;
          nativeBuildInputs = [ p.cmake p.ninja p.clang p.cudatoolkit ];
          configurePhase = ''
            # CMAKE_CXX_SCAN_FOR_MODULES=OFF: same CMake 4 clang-scan-deps
            # workaround as the host-only engine.
            cmake -B build -S . -G Ninja \
              -DGRANITE_ENABLE_CUDA=ON \
              -DCMAKE_CXX_SCAN_FOR_MODULES=OFF
          '';
          buildPhase = ''
            # Override stdenv's default buildPhase (which would run a bare
            # `make`); the Ninja build is driven through CMake instead.
            cmake --build build
          '';
          # Full sm_89 compile, including the device-test and bench binaries;
          # run every ctest suite except the GPU-only kernels_device run.
          checkPhase = ''
            ctest --test-dir build --output-on-failure \
              --exclude-regex 'kernels_device|verify_gemv_device'
          '';
          installPhase = ''
            mkdir -p $out/lib $out/bin $out/include/granite4
            install -m 0755 build/engine/libgranite4.${libext pkgs} $out/lib/
            install -m 0755 build/engine/granite4-checkpoint $out/bin/
            # CUDA-only artifacts: the device-correctness test (one-command
            # re-verification on the GPU box), the micro-benchmark, and the
            # inference CLI whose --gpu 1 path runs the sm_89 forward pass.
            install -m 0755 build/engine/granite4-kernel-test $out/bin/
            install -m 0755 build/engine/granite4-kernel-bench $out/bin/
            install -m 0755 build/engine/granite4-gemv-bench $out/bin/
            install -m 0755 build/engine/granite4-verify-gemv-test $out/bin/
            install -m 0755 build/engine/granite4-verify-gemv-bench $out/bin/
            install -m 0755 build/engine/granite4-infer $out/bin/
            install -m 0644 engine/src/*.h $out/include/granite4/
          '';
          meta.description = "Granite 4.2 8B engine with sm_89 CUDA kernels (CUDA on); device test re-run on the GPU box";
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
            # The parity gate ships with the frontend so a deployment can
            # (re)validate the engine it serves with; the teacher-forced GPU
            # mode against the shipped reference needs no torch.
            mkdir -p $out/lib/scripts
            cp scripts/parity_gate.py $out/lib/scripts/
            printf '%s\n' \
              '#!${interpreter}/bin/python3' \
              'import os, runpy, sys' \
              '_lib = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")' \
              'sys.argv[0] = "granite4-parity-gate"' \
              'runpy.run_path(os.path.join(_lib, "scripts", "parity_gate.py"), run_name="__main__")' \
              > $out/bin/granite4-parity-gate
            chmod 0755 $out/bin/granite4-parity-gate
          '';
          meta.description = "Granite 4.2 8B API frontend (OpenAI-style chat completions over the native engine)";
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
          # CUDA engine, Linux only. `optionalAttrs` keeps the attribute absent
          # (rather than null) on other systems, so `nix flake check` still
          # passes there; the GPU host references it on x86_64-linux only.
          // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux
            { engine-cuda = mkEngineCuda pkgs; }
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
      # services.granite4-rtx, passing the packages from that input. The
      # module never references this flake's self.
      #
      # Without `checkpoint` the service is the Phase-0 scaffold (validates
      # requests, 501 for completions). With `checkpoint` (and an
      # enginePackage), every start first runs the parity gate on the GPU
      # (granite4-parity-gate, teacher-forced against the shipped
      # transformers reference); only a passing gate artifact enables
      # serving, through one persistent `granite4-infer serve --gpu 1`
      # process that keeps the model resident. A failing gate leaves the API
      # up in its 501 state.
      nixosModules.granite4Rtx =
        { config, lib, pkgs, ... }:
        let
          cfg = config.services.granite4-rtx;
          serving = cfg.checkpoint != null;
          stateDir = "/var/lib/granite4-rtx";
          gateFile = "${stateDir}/parity_gate.json";
          gateCmd = lib.escapeShellArgs [
            "${cfg.package}/bin/granite4-parity-gate"
            "--checkpoint" cfg.checkpoint
            "--engine" "${cfg.enginePackage}/bin/granite4-infer"
            "--prompt" "The capital of France is"
            "--max-tokens" "24"
            "--gpu"
            "--mode" "teacher-forced"
            "--reference" "${cfg.package}/lib/reference/granite/parity_ref.json"
            "--ulp-tolerance" (toString cfg.gate.ulpTolerance)
            "--min-match" (toString cfg.gate.minMatch)
            "--gate-out" gateFile
            "--out" "${stateDir}/parity_report.json"
          ];
        in
        {
          options.services."granite4-rtx" = {
            enable = lib.mkEnableOption "the Granite 4.2 8B API service";
            package = lib.mkOption {
              description = "The granite4-api derivation (flake packages.<system>.api)";
              type = lib.types.package;
            };
            enginePackage = lib.mkOption {
              description = "The granite4-engine-cuda derivation (flake packages.<system>.engine-cuda); required when checkpoint is set";
              type = lib.types.nullOr lib.types.package;
              default = null;
            };
            checkpoint = lib.mkOption {
              description = "Granite 4.2 8B checkpoint directory (BF16 safetensors + tokenizer). null: Phase-0 scaffold, no inference.";
              type = lib.types.nullOr lib.types.str;
              default = null;
              example = "/opt/granite4-rtx-4070-ti-super/ckpt";
            };
            maxContext = lib.mkOption {
              description = "Positions (prompt + completion) per request for the GPU engine; its KV cache costs 160 KiB of VRAM per position.";
              type = lib.types.ints.positive;
              default = 8192;
            };
            eagle.head = lib.mkOption {
              description = "Trained EAGLE draft head (safetensors, scripts/eagle/export_head.py) for speculative decoding on the GPU engine. The engine's exact greedy verify keeps output identical to plain greedy decoding; null decodes without drafting.";
              type = lib.types.nullOr lib.types.str;
              default = null;
              example = "/opt/granite4-rtx-4070-ti-super/eagle/heads/r1.safetensors";
            };
            eagle.k = lib.mkOption {
              description = "Tokens the EAGLE head drafts per verify round (1-15).";
              type = lib.types.ints.between 1 15;
              default = 5;
            };
            gate.minMatch = lib.mkOption {
              description = "Fraction of reference positions that must pass the teacher-forced gate.";
              type = lib.types.numbers.between 0 1;
              default = 1.0;
            };
            gate.ulpTolerance = lib.mkOption {
              description = "bf16 ULPs the reference token's logit may sit below the engine's top logit (ties and bf16 rounding).";
              type = lib.types.ints.unsigned;
              default = 2;
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

          config = lib.mkIf cfg.enable {
            assertions = [
              {
                assertion = !serving || cfg.enginePackage != null;
                message = "services.granite4-rtx.checkpoint needs services.granite4-rtx.enginePackage (packages.<system>.engine-cuda)";
              }
            ];
            users = {
              groups."granite4-rtx" = { };
              users."granite4-rtx" = {
                isSystemUser = true;
                group = "granite4-rtx";
                description = "Granite 4.2 8B API";
              };
            };
            systemd.services.granite4-rtx = {
              description =
                if serving then "Granite 4.2 8B API (GPU engine, parity-gated)"
                else "Granite 4.2 8B API scaffold (Phase 0: validates requests, no inference)";
              after = [ "network.target" ] ++ lib.optional serving "nvidia-persistenced.service";
              wants = lib.optional serving "nvidia-persistenced.service";
              wantedBy = [ "multi-user.target" ];
              environment = lib.optionalAttrs serving ({
                GRANITE4_GATE = gateFile;
                GRANITE4_MAX_CONTEXT = toString cfg.maxContext;
              } // lib.optionalAttrs (cfg.eagle.head != null) {
                GRANITE4_EAGLE_HEAD = cfg.eagle.head;
                GRANITE4_EAGLE_K = toString cfg.eagle.k;
              });
              serviceConfig = {
                User = "granite4-rtx";
                Group = "granite4-rtx";
                ExecStart = "${cfg.package}/bin/granite4-api --host ${cfg.listenAddress} --port ${toString cfg.listenPort}";
                Restart = "on-failure";
                RestartSec = 5;
                NoNewPrivileges = true;
                ProtectSystem = "strict";
                ProtectHome = true;
                PrivateTmp = true;
                RestrictAddressFamilies = [ "AF_INET" "AF_INET6" "AF_UNIX" ];
                CapabilityBoundingSet = "";
              } // (if serving then {
                StateDirectory = "granite4-rtx";
                # A stale artifact must never enable serving for a new engine
                # or checkpoint: drop it, then re-run the gate ("-": a failing
                # gate leaves the API up in its 501 state).
                ExecStartPre = [
                  "${pkgs.coreutils}/bin/rm -f ${gateFile}"
                  "-${gateCmd}"
                ];
                TimeoutStartSec = 900;
                # The engine needs the NVIDIA device nodes and the host
                # driver libraries (/run/opengl-driver, read-only).
                PrivateDevices = false;
                DevicePolicy = "closed";
                DeviceAllow = [
                  "/dev/nvidiactl rw"
                  "/dev/nvidia0 rw"
                  "/dev/nvidia-uvm rw"
                  "/dev/nvidia-uvm-tools rw"
                  "/dev/nvidia-modeset rw"
                  "char-nvidia-caps r"
                ];
                LimitMEMLOCK = "infinity";
              } else {
                # The Phase 0 process only opens a listening socket.
                PrivateDevices = true;
              });
            };
          };
        };
    };

}
