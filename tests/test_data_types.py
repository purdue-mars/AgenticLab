from agenticlab.core.data_types import (
    CheckResult,
    DetectionResult,
    GraspPose,
    PlacingPoint,
    TaskPlan,
)


def test_detection_result_round_trip():
    obj = DetectionResult(
        success=True,
        objects=[{"bbox": [1, 2, 3, 4], "label": "cup", "score": 0.9}],
        image_shape=(480, 640),
        raw_output={"raw": "ok"},
        summary={"n": 1},
    )
    d = obj.to_dict()
    obj2 = DetectionResult(**d)
    assert obj2.to_dict() == d


def test_check_result_round_trip():
    obj = CheckResult(
        success=False,
        reasoning="missing precondition",
        timestamp="2026-04-22T00:00:00Z",
        elapsed_time=0.123,
        failed_conditions=["hand_empty"],
        detailed_analysis=[{"cond": "hand_empty", "ok": False}],
        token_usage=42,
    )
    d = obj.to_dict()
    obj2 = CheckResult(**d)
    assert obj2.to_dict() == d


def test_grasp_pose_round_trip():
    obj = GraspPose(
        translation=[0.1, 0.2, 0.3],
        rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        width=0.05,
        evaluation={"ok": True},
        visualization_path=None,
    )
    d = obj.to_dict()
    obj2 = GraspPose(**d)
    assert obj2.to_dict() == d


def test_placing_point_round_trip():
    obj = PlacingPoint(
        point=[123, 456],
        pose_3d=[0.1, 0.2, 0.3],
        evaluation=[{"iter": 1, "ok": True}],
        visualization_path="tmp.png",
        yaw_rotation=15.0,
        roll_rotation=-5.0,
        place_offset=0.05,
        orientation_reasoning="stable",
    )
    d = obj.to_dict()
    obj2 = PlacingPoint(**d)
    assert obj2.to_dict() == d


def test_task_plan_round_trip():
    obj = TaskPlan(
        task_description="pick up the cup",
        objects_identified=["cup"],
        reasoning="simple pick",
        action_sequence=["pick(cup)"],
        elapsed_time=0.456,
        timestamp="2026-04-22T00:00:00Z",
        updated_domain=None,
        problem_pddl=None,
        action_details=[{"action": "pick", "object": "cup"}],
        goal_conditions=["holding(cup)"],
        token_usage=7,
    )
    d = obj.to_dict()
    obj2 = TaskPlan(**d)
    assert obj2.to_dict() == d
