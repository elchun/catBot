# partly based on https://github.com/RussTedrake/manipulation/blob/master/manipulation/envs/box_flipup.py
import numpy as np
import gym
from IPython.display import HTML, SVG, display
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    PidController,
    MeshcatVisualizer,
    MultibodyPlant,
    Parser,
    Simulator,
    StartMeshcat,
    PassThrough,
    Multiplexer,
    ConstantVectorSource,
    LeafSystem,
    Variable,
    EventStatus,
    RandomGenerator,
)

from catbot.utils.drake_gym import DrakeGymEnv

from catbot.utils.meshcat_util import MeshcatCatBotSliders

def make_catbot_env(generator,
                    observations="state",
                    meshcat=None,
                    time_limit=4):
    time_step = 1e-3

    builder = DiagramBuilder()

    # -- Add original plant -- #
    # Adds both MultibodyPlant and the SceneGraph, and wires them together.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    # Note that we parse into both the plant and the scene_graph here.
    model = Parser(plant, scene_graph).AddModelFromFile(
        "../models/singleAxisCatBot.urdf")

    gravity_field = plant.mutable_gravity_field()
    gravity_field.set_gravity_vector(np.array([0.0, 0.0, 0.0]))

    plant.Finalize()

    # print('plant names: ', plant.GetPositionNames(model))
    # print('plant state: ', plant.GetState(model))

    # -- Add controller plant -- #
    controller_plant = MultibodyPlant(time_step=time_step)
    model = Parser(controller_plant).AddModelFromFile(
        "../models/singleAxisCatBot.urdf")
    controller_gravity_field = controller_plant.mutable_gravity_field()
    controller_gravity_field.set_gravity_vector(np.array([0.0, 0.0, 0.0]))
    controller_plant.Finalize()

    # -- Add visualizer -- #
    if meshcat:
        visualizer = MeshcatVisualizer.AddToBuilder(builder,
                                            scene_graph.get_query_output_port(),
                                            meshcat)
        meshcat.ResetRenderMode()
        meshcat.DeleteAddedControls()

    # -- Add controller -- #

    # actuator pos is a_rev, b_rev, a_hinge, b_hinge
    # state is Center, a_hinge?, a_rot, b_hinge, b_rot
    # idk why they're different
    state_projection_matrix = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],

        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    ])

    N = 4  # Num controller states

    kp = [0.03, 0.03, 0.03, 0.03]
    ki = [0.00, 0.00, 0.0, 0.0]
    kd = [0.025, 0.025, 0.04, 0.04]
    catbot_controller = builder.AddSystem(
        PidController(state_projection_matrix, kp, ki, kd)
    )
    catbot_controller.set_name("catbot_controller")

    # Set up estimated state input to controller
    builder.Connect(
        plant.get_state_output_port(model),
        catbot_controller.get_input_port_estimated_state(),
    )

    # Set up controller desired state
    actions = builder.AddSystem(PassThrough(N))
    positions_to_state = builder.AddSystem(Multiplexer([N, N]))

    builder.Connect(
        actions.get_output_port(),
        positions_to_state.get_input_port(0))

    zeros = builder.AddSystem(ConstantVectorSource(np.zeros(N)))
    builder.Connect(zeros.get_output_port(),
                    positions_to_state.get_input_port(1))

    builder.Connect(
        positions_to_state.get_output_port(),
        catbot_controller.get_input_port_desired_state()
    )

    # Connect controller output
    builder.Connect(catbot_controller.get_output_port(),
                    plant.get_actuation_input_port())


    # -- Export ports -- #
    builder.ExportInput(actions.get_input_port(), "actions")

    class ConvertStateDtype(LeafSystem):
        def __init__(self):
            LeafSystem.__init__(self)
            self.DeclareVectorInputPort("state_in", 10)
            self.DeclareVectorOutputPort("state_out", 10, self.convert)

        def convert(self, context, output):
            state = self.get_input_port().Eval(context)
            float_state = np.array(state, dtype=np.float32)
            for i in range(10):
                output[i] = float_state[i]


    if observations == "state":
        # convert_state_dtype = builder.AddSystem(ConvertStateDtype())
        # builder.Connect(plant.get_state_output_port(), convert_state_dtype.get_input_port())
        # builder.ExportOutput(convert_state_dtype.get_output_port(), "observations")
        builder.ExportOutput(plant.get_state_output_port(), "observations")
    else:
        raise ValueError("Observations must be 'state'")

    # -- Add Reward -- #
    class RewardSystem(LeafSystem):
        def __init__(self):
            LeafSystem.__init__(self)
            self.DeclareVectorInputPort("catbot_state", 10)
            self.DeclareVectorInputPort("actions", 4)
            self.DeclareVectorOutputPort("reward", 1, self.CalcReward)

        def CalcReward(self, context, output):
            # state is Center, a_hinge, a_rot, b_hinge, b_rot
            catbot_state = self.get_input_port(0).Eval(context)
            actions = self.get_input_port(1).Eval(context)


            # So we clamp the angle between 0 and 2pi
            a_rot_world = (catbot_state[0] + catbot_state[2]) % (np.pi * 2) - np.pi
            b_rot_world = (catbot_state[0] + catbot_state[4]) % (np.pi * 2) - np.pi
            center_from_vertical = (catbot_state[0] % (2 * np.pi)) - np.pi

            a_hinge = catbot_state[1]
            b_hinge = catbot_state[3]

            # Add position cost
            # -- COST 1 -- #
            # cost = a_rot_world**2 + \
            #     b_rot_world**2 + \
            #     2 * center_from_vertical**2

            # -- COST 2 -- #
            # Want a and b hinge world to have the same sign and face down
            # cost = -a_rot_world * b_rot_world + center_from_vertical**2

            # -- COST 3 -- #
            # Center cost?
            # cost = center_from_vertical**2

            # # -- COST 4 -- #
            # cost = a_rot_world**2 + \
            #     b_rot_world**2

            # -- COST 5 -- #
            # Want both feet to be facing the same direction
            # cost = (a_rot_world**2 + b_rot_world**2) + 5 * np.sign(-a_rot_world * b_rot_world)

            # -- COST 6 -- #
            # Simple cost
            # cost = (a_rot_world**2 + b_rot_world**2)

            # -- COST 7 -- #
            # cost = a_rot_world**2 + b_rot_world**2 + center_from_vertical**2

            # -- COST 8 -- #
            # (Sparese cost)
            # if a_rot_world ** 2 > (np.pi / 4)**2 or b_rot_world ** 2 > (np.pi / 4)**2:
            #     cost = 1
            # else:
            #     cost = 0

            # -- COST 9 -- #
            # Normal with end reward

            # cost = (a_rot_world**2 + b_rot_world**2)
            # if a_rot_world ** 2 < (np.pi / 8)**2 and b_rot_world ** 2 < (np.pi / 8)**2:
            #     cost -= 100

            # # -- COST 10 -- #
            # # Add cost so hinges are the same
            # cost = a_rot_world**2 + b_rot_world**2 + 10 * (a_hinge - b_hinge)**2

            # # Add final position cost
            # if a_rot_world ** 2 < (np.pi / 8)**2 and b_rot_world ** 2 < (np.pi / 8)**2:
            #     cost -= 100

            # # -- COST 11 -- #
            # cost = a_rot_world**2 + b_rot_world**2 \
            #     + center_from_vertical**2 + 10 * (a_hinge - b_hinge)**2

            # # Add final position cost
            # if a_rot_world ** 2 < (np.pi / 8)**2 and b_rot_world ** 2 < (np.pi / 8)**2:
            #     cost -= 100

            # -- COST 12 -- #
            # cost = a_rot_world**2 + b_rot_world**2 \
            #     + 10 * (a_hinge - b_hinge)**2 \
            #     + 0.2 * (a_hinge ** 2 + b_hinge ** 2)
            #     # + center_from_vertical**2 \

            # -- COST 13 -- #
            # Sparse reward
            # Add final position cost
            if a_rot_world ** 2 < (np.pi / 8)**2 and b_rot_world ** 2 < (np.pi / 8)**2:
                cost = -100
            else:
                cost = 0

            # -- Effort cost -- #
            # state_to_control_projection = np.array([
            #     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            #     [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            #     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            #     [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            # ])

            # # Add effort cost
            # effort = actions - state_to_control_projection.dot(catbot_state)
            # cost += 0.1 * effort.dot(effort)

            # Add to make reward positive (to avoid rewarding simulator crashes)
            # Quote from manipulation repo
            # output[0] = 30 - cost

            # -- Convert cost to reward -- #
            output[0] = -cost

    reward = builder.AddSystem(RewardSystem())
    builder.Connect(plant.get_state_output_port(model), reward.get_input_port(0))
    builder.Connect(actions.get_output_port(), reward.get_input_port(1))
    builder.ExportOutput(reward.get_output_port(), "reward")

    # -- Set random state distributions -- #
    center_uniform_random = Variable(
        name="uniform_random", type=Variable.Type.RANDOM_UNIFORM
    )

    a_hinge_uniform_random = Variable(
        name="uniform_random", type=Variable.Type.RANDOM_UNIFORM
    )

    b_hinge_uniform_random = Variable(
        name="uniform_random", type=Variable.Type.RANDOM_UNIFORM
    )

    a_rev_uniform_random = Variable(
        name="uniform_random", type=Variable.Type.RANDOM_UNIFORM
    )

    b_rev_uniform_random = Variable(
        name="uniform_random", type=Variable.Type.RANDOM_UNIFORM
    )

    plant.GetJointByName("hinge_revolute").set_random_angle_distribution(
        # 2 * np.pi * center_uniform_random - np.pi)
        np.pi/2 * center_uniform_random - np.pi / 4)

    plant.GetJointByName("A_hinge").set_random_angle_distribution(
        (2 * np.pi/3) * a_hinge_uniform_random - np.pi/3)

    plant.GetJointByName("B_hinge").set_random_angle_distribution(
        (2 * np.pi/3) * b_hinge_uniform_random - np.pi/3)

    plant.GetJointByName("A_revolute").set_random_angle_distribution(
        (np.pi) * a_hinge_uniform_random - np.pi/2)

    plant.GetJointByName("B_revolute").set_random_angle_distribution(
        (np.pi) * b_hinge_uniform_random - np.pi/2)


    # -- Build diagram and sim -- #
    diagram = builder.Build()
    simulator = Simulator(diagram)

    def monitor(context):
        if context.get_time() > time_limit:
            return EventStatus.ReachedTermination(diagram, "time limit")
        return EventStatus.Succeeded()

    simulator.set_monitor(monitor)
    return simulator


def CatBotEnv(observations="state", meshcat=None, time_limit=10):
    simulator = make_catbot_env(
        RandomGenerator(),
        observations=observations,
        meshcat=meshcat,
        time_limit=time_limit)

    action_space = gym.spaces.Box(
        low=np.array([-np.pi/2, -np.pi/2, -np.pi/3, -np.pi/3]),
        high=np.array([np.pi/2, np.pi/2, np.pi/3, np.pi/3]),
        dtype=np.float32)

    plant = simulator.get_system().GetSubsystemByName("plant")

    # Quote from https://github.com/RussTedrake/manipulation/blob/master/manipulation/envs/box_flipup.py#L17
    # It is unsound to use raw MBP dof limits as observation bounds, because
    # most simulations will violate those limits in practice (in collision, or
    # due to gravity, or in general because no constraint force is incurred
    # except in violation).  However we don't have any other better limits
    # here.  So we broaden the limits by a fixed offset and hope for the best.

    NUM_DOFS = 5
    POSITION_LIMIT_TOLERANCE = np.full((NUM_DOFS,), 0.1)
    VELOCITY_LIMIT_TOLERANCE = np.full((NUM_DOFS,), 0.5)
    if observations == "state":
        low = np.concatenate(
            (
                plant.GetPositionLowerLimits() - POSITION_LIMIT_TOLERANCE,
                plant.GetVelocityLowerLimits() - VELOCITY_LIMIT_TOLERANCE,
            )
        )
        high = np.concatenate(
            (
                plant.GetPositionUpperLimits() + POSITION_LIMIT_TOLERANCE,
                plant.GetVelocityUpperLimits() + VELOCITY_LIMIT_TOLERANCE,
            )
        )
        observation_space = gym.spaces.Box(
            low=np.asarray(low), high=np.asarray(high), dtype=np.float32
        )

    env = DrakeGymEnv(
        simulator=simulator,
        time_step=0.1,
        action_space=action_space,
        observation_space=observation_space,
        reward="reward",
        action_port_id="actions",
        observation_port_id="observations",
    )

    return env