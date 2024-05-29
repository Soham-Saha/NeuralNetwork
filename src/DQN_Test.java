//This test doesn't work as of now

import java.util.ArrayList;
import java.util.Collections;

public class DQN_Test {

	static State currentState;
	static Sequential tmpNet;
	static Sequential targetNet;
	static Trainer trainer;// Will train the tmpNet
	static ArrayList<Experience> experienceReplayBuffer = new ArrayList<>();
	static double episode;

	static double maxEpisodes = 100000;
	static double learningRate = 1e-7;
	static double discountFactor = 0.95;
	static double minEpsilon = 0.01;
	static int miniBatchSize = 32;
	static int minimumBufferSize = 500;
	static int maxTrainIteration = 1;

	static double maxHeight = 300;
	static double initDistance = 400;
	static int permissibleLimit = 3;
	static int fallRate = 2;
	static int boostRate = 4;
	static int barrier;

	public static void main(String[] args) throws CloneNotSupportedException, InterruptedException {
		init();
		startTraining();
	}

	private static void startTraining() throws InterruptedException, CloneNotSupportedException {
		double startEp = 0.9;
		double endEp = 0.5;
		episode = 0;
		while (episode < maxEpisodes) {
			currentState = generateInitialState();
			double ep = (startEp - endEp) * (maxEpisodes - episode) / maxEpisodes + endEp;
			// System.out.println("Episode: " + episode);
			// System.out.print(currentState + " -> ");
			while (true) {
				Action act = chooseActionForState(currentState, ep);
				State next = nextState(currentState, act);
				experienceReplayBuffer.add(new Experience(currentState, act, reward(currentState, act), next, isEndState(next)));
				currentState = next;
				// System.out.print(currentState + " -> ");
				if (isEndState(next)) {
					break;
				}
			}
			// System.out.println();
			episode++;
			if (experienceReplayBuffer.size() > minimumBufferSize) {
				System.out.println();
				train();
			}
			if (episode % 100 == 0) {
				targetNet = (Sequential) tmpNet.clone();
			}
		}
	}

	private static void train() throws InterruptedException {
		Experience[] miniExperienceBatch = new Experience[miniBatchSize];
		Collections.shuffle(experienceReplayBuffer);
		for (int i = 0; i < miniBatchSize; i++) {
			miniExperienceBatch[i] = experienceReplayBuffer.remove(experienceReplayBuffer.size() - 1);
		}
		DataSet ds = new DataSet(miniBatchSize);
		for (int i = 0; i < miniBatchSize; i++) {
			Experience exp = miniExperienceBatch[i];
			double[] y = { 0 };
			if (exp.done) {
				y[0] = exp.reward;
			} else {
				y[0] = exp.reward + discountFactor * Math.max(targetNet.predict(encode(exp.nextState, Action.PRESS))[0], targetNet.predict(encode(exp.nextState, Action.IDLE))[0]);
			}
			ds.records[i] = new Record(encode(exp.state, exp.action), y);
			System.out.println(targetNet.predict(encode(new State(10, 50, 60), Action.PRESS))[0] + " " + targetNet.predict(encode(new State(10, 50, 60), Action.IDLE))[0]);
		}
		trainer.fit(ds, maxTrainIteration);
	}

	public static double reward(State state, Action action) {
		State next = nextState(state, action);
		if (next.heightFromGround < 0 || next.heightFromGround > maxHeight) {
			return -500;
		}
		if (isEndState(next)) {
			if (Math.abs(next.heightFromHole) <= permissibleLimit) {
				return 500;
			}
			return -500;
		}
		return 100;
	}

	public static Action chooseActionForState(State state, double ep) {
		double k = Math.random();
		if (k > ep) {
			// Exploit
			return targetNet.predict(encode(state, Action.PRESS))[0] > targetNet.predict(encode(state, Action.IDLE))[0] ? Action.PRESS : Action.IDLE;
		} else {
			// Explore
			return k > 0.5 ? Action.PRESS : Action.IDLE;
		}
	}

	public static State generateInitialState() {
		barrier = (int) (Math.random() * (maxHeight - 2 * permissibleLimit)) + permissibleLimit;
		double heightFromGround = (int) (Math.random() * maxHeight);
		return new State(initDistance - 1, heightFromGround - barrier, heightFromGround);
	}

	public static State nextState(State s, Action a) {
		int k = (a.equals(Action.PRESS) ? boostRate : 0) - fallRate;
		return new State(s.distance - 1, s.heightFromHole + k, s.heightFromGround + k);
	}

	public static boolean isEndState(State s) {
		return s.distance == -1 || s.heightFromGround < 0 || s.heightFromGround > maxHeight;
	}

	private static void init() throws CloneNotSupportedException {
		tmpNet = new Sequential(3 + 2, 10, 1);
		// tmpNet.layers[tmpNet.layers.length - 1].activation =
		// ActivationFunctions.get(ActivationFunctions.TANH);
		targetNet = (Sequential) tmpNet.clone();
		trainer = new Trainer(tmpNet, ErrorFunctions.MEAN_SQUARED_ERROR, learningRate) {
			@Override
			public void perEpoch(int time, double error) throws InterruptedException {
				if (time == 1) {
					System.out.println(episode + "\t" + error);
				}
			}
		};
	}

	public static double[] encode(State state, Action action) {
		return new double[] { state.distance, state.heightFromGround, state.heightFromHole, action.equals(Action.PRESS) ? 1 : 0, action.equals(Action.IDLE) ? 1 : 0 };
	}

}

class State {

	double distance;
	double heightFromHole;
	double heightFromGround;

	public State(double distance, double heightFromHole, double heightFromGround) {
		this.distance = distance;
		this.heightFromHole = heightFromHole;
		this.heightFromGround = heightFromGround;
	}

	@Override
	public String toString() {
		return distance + " " + heightFromHole + " " + heightFromGround;
	}
}

enum Action {
	PRESS, IDLE;
}

class Experience {
	State state;
	Action action;
	double reward;
	State nextState;
	boolean done; // Whether this 'action' ended the episode

	public Experience(State state, Action action, double reward, State nextState, boolean done) {
		this.state = state;
		this.action = action;
		this.reward = reward;
		this.nextState = nextState;
		this.done = done;
	}

	@Override
	public String toString() {
		return "(" + state + ", " + action + ", " + reward + ", " + nextState + ", " + done + ")";
	}
}