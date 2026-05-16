def test_import_obj_detector():
    import agenticlab.core.perception.obj_detector  # noqa: F401


def test_obj_detector_unknown_backend_raises():
    from agenticlab.core.perception.obj_detector import ObjectDetector

    detector = ObjectDetector({"ObjDetector": {"method": "doesnotexist"}})
    try:
        detector.detect(None, "x")  # type: ignore[arg-type]
    except ValueError as e:
        assert "Unsupported detection method" in str(e)
