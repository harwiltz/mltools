{ lib
, buildPythonPackage
, numpy
}:

buildPythonPackage rec {
  pname = "mltools";
  version = "0.0.1";
  src = ./.;
  pythonImportsCheck = [ "mltools" ];
  # Currently requires comet_ml, not FOSS
  doCheck = false;
  meta = with lib; {
    license = licenses.mit;
  };
}
