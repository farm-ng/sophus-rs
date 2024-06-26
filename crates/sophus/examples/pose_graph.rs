use hollywood::actors::egui::EguiActor;
use hollywood::actors::egui::Stream;
use hollywood::prelude::*;
use sophus::opt::example_problems::pose_circle::PoseCircleProblem;
use sophus::prelude::*;
use sophus::viewer::actor::run_viewer_on_main_thread;
use sophus::viewer::actor::ViewerBuilder;
use sophus::viewer::actor::ViewerCamera;
use sophus::viewer::actor::ViewerConfig;
use sophus::viewer::renderables::*;
use sophus_core::linalg::VecF64;
use sophus_image::ImageSize;
use sophus_lie::Isometry2;
use sophus_lie::Isometry3;
use sophus_sensor::camera_enum::perspective_camera::PerspectiveCameraEnum;
use sophus_sensor::dyn_camera::DynCamera;
use sophus_sensor::KannalaBrandtCamera;
use sophus_viewer::interactions::WgpuClippingPlanes;
use sophus_viewer::simple_viewer::SimpleViewer;

use crate::color::Color;
use crate::renderable3d::Line3;
use crate::renderable3d::Lines3;
use crate::renderable3d::Renderable3d;
use crate::renderable3d::View3dPacket;

#[actor(ContentGeneratorMessage, NullInRequestMessage)]
type ContentGenerator = Actor<
    NullProp,
    ContentGeneratorInbound,
    NullInRequests,
    ContentGeneratorState,
    ContentGeneratorOutbound,
    NullOutRequests,
>;

/// Inbound message for the ContentGenerator actor.
#[derive(Clone, Debug)]
#[actor_inputs(
    ContentGeneratorInbound,
    {
        NullProp,
        ContentGeneratorState,
        ContentGeneratorOutbound,
        NullOutRequests,
        NullInRequestMessage})]
pub enum ContentGeneratorMessage {
    /// in seconds
    ClockTick(f64),
}

#[derive(Clone, Debug, Default)]
pub struct ContentGeneratorState {}

/// Outbound hub for the ContentGenerator.
#[actor_outputs]
pub struct ContentGeneratorOutbound {
    /// curves
    pub packets: OutboundChannel<Stream<Packets>>,
}

fn make_axes(world_from_local_poses: Vec<Isometry2<f64, 1>>) -> Vec<Line3> {
    let zero_in_local = VecF64::<2>::zeros();
    let x_axis_local = VecF64::<2>::new(1.0, 0.0);
    let y_axis_local = VecF64::<2>::new(0.0, 1.0);

    let mut lines = vec![];

    for world_from_local in world_from_local_poses.iter() {
        let zero_in_world = world_from_local.transform(&zero_in_local);
        let axis_x_in_world = world_from_local.transform(&x_axis_local);
        let axis_y_in_world = world_from_local.transform(&y_axis_local);

        lines.push(Line3 {
            p0: VecF64::<3>::new(zero_in_world.x, zero_in_world.y, 0.0).cast(),
            p1: VecF64::<3>::new(axis_x_in_world.x, axis_x_in_world.y, 0.0).cast(),
            color: Color {
                r: 1.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
            line_width: 2.0,
        });
        lines.push(Line3 {
            p0: VecF64::<3>::new(zero_in_world.x, zero_in_world.y, 0.0).cast(),
            p1: VecF64::<3>::new(axis_y_in_world.x, axis_y_in_world.y, 0.0).cast(),
            color: Color {
                r: 0.0,
                g: 1.0,
                b: 0.0,
                a: 1.0,
            },
            line_width: 2.0,
        });
    }

    lines
}

impl HasOnMessage for ContentGeneratorMessage {
    /// Process the inbound time_stamp message.
    fn on_message(
        self,
        _prop: &Self::Prop,
        _state: &mut Self::State,
        outbound: &Self::OutboundHub,
        _request: &NullOutRequests,
    ) {
        match &self {
            ContentGeneratorMessage::ClockTick(_time_in_seconds) => {
                let pose_graph = PoseCircleProblem::new(25);

                // Camera / view pose parameters
                let intrinsics = KannalaBrandtCamera::<f64, 1>::new(
                    &VecF64::<8>::from_array([600.0, 600.0, 320.0, 240.0, 0.0, 0.0, 0.0, 0.0]),
                    ImageSize {
                        width: 640,
                        height: 480,
                    },
                );
                let scene_from_camera = Isometry3::from_t(&VecF64::<3>::new(0.0, 0.0, -50.0));
                let clipping_planes = WgpuClippingPlanes {
                    near: 0.1,
                    far: 1000.0,
                };

                let initial_camera = ViewerCamera {
                    intrinsics: DynCamera::<f64, 1>::from_model(
                        PerspectiveCameraEnum::KannalaBrandt(intrinsics),
                    ),
                    clipping_planes,
                    scene_from_camera,
                };

                let mut packet = View3dPacket {
                    view_label: "view0".to_owned(),
                    renderables3d: vec![],
                    initial_camera,
                };

                packet.renderables3d = vec![
                    Renderable3d::Lines3(Lines3 {
                        name: "true".into(),
                        lines: make_axes(pose_graph.true_world_from_robot.clone()),
                    }),
                    Renderable3d::Lines3(Lines3 {
                        name: "est".into(),
                        lines: make_axes(pose_graph.est_world_from_robot.clone()),
                    }),
                ];

                outbound.packets.send(Stream {
                    msg: Packets {
                        packets: vec![Packet::View3d(packet)],
                    },
                });
            }
        }
    }
}

impl IsInboundMessageNew<f64> for ContentGeneratorMessage {
    fn new(_inbound_name: String, msg: f64) -> Self {
        ContentGeneratorMessage::ClockTick(msg)
    }
}

pub async fn run_viewer_example() {
    let mut builder = ViewerBuilder::from_config(ViewerConfig {});

    // Pipeline configuration
    let pipeline = Hollywood::configure(&mut |context| {
        // Actor creation:
        // 1. Periodic timer to drive the simulation
        let mut timer = hollywood::actors::Periodic::new_with_period(context, 0.01);
        // 2. The content generator of the example
        let mut content_generator = ContentGenerator::from_prop_and_state(
            context,
            NullProp {},
            ContentGeneratorState::default(),
        );
        // 3. The viewer actor
        let mut viewer = EguiActor::<Packets, String, Isometry3<f64, 1>, (), ()>::from_builder(
            context, &builder,
        );

        // Pipeline connections:
        timer
            .outbound
            .time_stamp
            .connect(context, &mut content_generator.inbound.clock_tick);
        content_generator
            .outbound
            .packets
            .connect(context, &mut viewer.inbound.stream);
    });

    // The cancel_requester is used to cancel the pipeline.
    builder
        .cancel_request_sender
        .clone_from(&pipeline.cancel_request_sender_template);

    // Plot the pipeline graph to the console.
    pipeline.print_flow_graph();

    // Pipeline execution:

    // 1. Run the pipeline on a separate thread.
    let pipeline_handle = tokio::spawn(pipeline.run());
    // 2. Run the viewer on the main thread. This is a blocking call.
    run_viewer_on_main_thread::<ViewerBuilder, SimpleViewer>(builder);
    // 3. Wait for the pipeline to finish.
    pipeline_handle.await.unwrap();
}

fn main() {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async {
            run_viewer_example().await;
        })
}
