package itml.agents;

import itml.cards.Card;
import itml.cards.CardRest;
import itml.simulator.CardDeck;
import itml.simulator.StateAgent;
import itml.simulator.StateBattle;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import java.util.ArrayList;


/**
 *  This agent will demolish it's opponents. Not even the chosen one (Neo), can stand up to it.
 *
 *  @authors        Arnar Freyr Bjarnason
 *                  Kjartan Valur Kjartansson
 *                  Jón Gísli Björgvinsson
 */
public class MyAgent extends Agent{
    private int m_noThisAgent;     // Index of our agent (0 or 1).
    private int m_noOpponentAgent; // Index of opponent's agent.
    private Classifier classifier_;
    private Instances dataset;
    boolean learned;

    public MyAgent ( CardDeck deck, int msConstruct, int msPerMove, int msLearn){
        super(deck, msConstruct, msPerMove, msLearn);
        classifier_ = new J48();
        learned = false;
    }

    @Override
    public void startGame( int noThisAgent, StateBattle stateBattle ){
        m_noThisAgent = noThisAgent;
        m_noOpponentAgent = (noThisAgent == 0) ? 1 : 0;

    }

    @Override
    public void endGame( StateBattle stateBattle, double [] results ){
        //System.out.println("===");
    }

    @Override
    public Card act( StateBattle stateBattle ){
        if (!learned) return getStupidMove(stateBattle);

        double[] values = new double[8];
        StateAgent a = stateBattle.getAgentState(0);
        StateAgent o = stateBattle.getAgentState(1);
        values[0] = a.getCol();
        values[1] = a.getRow();
        values[2] = a.getHealthPoints();
        values[3] = a.getStaminaPoints();
        values[4] = o.getCol();
        values[5] = o.getRow();
        values[6] = o.getHealthPoints();
        values[7] = o.getStaminaPoints();
        try {
            ArrayList<Card> allCards = m_deck.getCards();
            ArrayList<Card> cards = m_deck.getCards(a.getStaminaPoints());
            Instance i = new Instance(1.0, values.clone());
            i.setDataset(dataset);
            int out = (int)classifier_.classifyInstance(i);
            Card selected = allCards.get(out);
            if(cards.contains(selected)) {
                return getLearnedMove(stateBattle, selected);
            }
        } catch (Exception e) {
            System.out.println("Error classifying new instance: " + e.toString());
        }
        return getStupidMove(stateBattle);
    }

    @Override
    public Classifier learn( Instances instances ){
        learned = true;
        dataset = instances;
        try {
            classifier_.buildClassifier(instances);
        } catch (Exception e){
            System.out.println("Error training classifier: " + e.toString());
        }
        return null;
    }

    /**
     * A function that returns the attack card with the highest damage available
     * @param stateBattle       Current state of game
     * @return attack card, null if none found
     */
    private Card attack( StateBattle stateBattle){
        StateAgent agent = stateBattle.getAgentState( m_noThisAgent );
        StateAgent oAgent  = stateBattle.getAgentState(m_noOpponentAgent);

        ArrayList<Card> cards = m_deck.getCards(agent.getStaminaPoints());
        Card bestCard = null;

        int bestHP = oAgent.getHealthPoints();
        // Find attack cards
        for ( Card card : cards ) {
            // In attack range?
            if ( (card.getType() == Card.CardActionType.ctAttack) &&
                    card.inAttackRange(agent.getCol(), agent.getRow(), oAgent.getCol(), oAgent.getRow()) ) {
                int currHP = oAgent.getHealthPoints() - card.getHitPoints();
                if (currHP <= 0) return card; // Killmove?
                // Find strongest attack card
                if(currHP < bestHP){
                    bestHP = currHP;
                    bestCard = card;
                }
            }
        }

        return bestCard;
    }

    /**
     * Will return the card that brings the agent as far from the opponent as possible
     *
     * @param stateBattle       Current state of the game
     * @param oCard             The predicted enemy card / null
     * @return a card moving the agent away from the opponent
     */
    private Card moveFurther (StateBattle stateBattle, Card oCard){
        Card [] move = new Card[2];
        if(learned) move[m_noOpponentAgent] = oCard;    // Predict where opponent moves
        else move[m_noOpponentAgent] = new CardRest();  // Assume opponent will remain still

        Card bestCard = new CardRest();
        int  minDistance = calcDistanceBetweenAgents( stateBattle );

        ArrayList<Card> cards = m_deck.getCards( stateBattle.getAgentState( m_noThisAgent ).getStaminaPoints() );
        // Move further from opponent
        for ( Card card : cards ) {
            StateBattle bs = (StateBattle) stateBattle.clone();   // clone the state, as play( ) modifies it.
            move[m_noThisAgent] = card;
            bs.play( move );
            int  distance = calcDistanceBetweenAgents( bs );
            if ( distance > minDistance ) {
                bestCard = card;
                minDistance = distance;
            }
        }
        return bestCard;
    }

    /**
     * Will return a card that brings the agent as close to the opponent as possible
     *
     * @param stateBattle       Current state of the game
     * @param oCard             The predicted enemy card / null
     * @return a card moving agent closer to opponent
     */
    private Card moveCloser( StateBattle stateBattle, Card oCard){
        Card [] move = new Card[2];
        if(learned) move[m_noOpponentAgent] = oCard;    // Predict where opponent moves
        else move[m_noOpponentAgent] = new CardRest();  // Assume opponent will remain still

        Card bestCard = new CardRest();
        int  bestDistance = calcDistanceBetweenAgents( stateBattle );

        ArrayList<Card> cards = m_deck.getCards(stateBattle.getAgentState( m_noThisAgent ).getStaminaPoints());
        // Move closer to the opponent.
        for ( Card card : cards ) {
            StateBattle bs = (StateBattle) stateBattle.clone();   // clone the state, as play( ) modifies it.
            move[m_noThisAgent] = card;
            bs.play( move );
            int  distance = calcDistanceBetweenAgents( bs );
            if ( distance < bestDistance ) {
                bestCard = card;
                bestDistance = distance;
            }
        }
        return bestCard;
    }

    /**
     * Gets a move, where the next card of the opponent is not predicted
     *
     * @param stateBattle       Current state of game
     * @return card with no prediction
     */
    private Card getStupidMove(StateBattle stateBattle){
        StateAgent agent = stateBattle.getAgentState(m_noThisAgent);
        StateAgent oAgent = stateBattle.getAgentState(m_noOpponentAgent);

        if(agent.getHealthPoints() >= oAgent.getHealthPoints()){
            Card c;
            if((c = attack(stateBattle)) != null) return c;
            else return moveCloser(stateBattle, null);
        }
        else{
            return moveFurther(stateBattle, null);
        }
    }

    /**
     * Get our next move, based on that the opponents move is oCard
     *
     * @param stateBattle       Current state of game
     * @param oCard             Predicted opponent move
     *
     * @return
     */

    private Card getLearnedMove(StateBattle stateBattle, Card oCard){
        StateAgent agent = stateBattle.getAgentState(m_noThisAgent);
        StateAgent oAgent = stateBattle.getAgentState(m_noOpponentAgent);
        //System.out.println("HP: " + agent.getHealthPoints() + ", EHP: " + oAgent.getHealthPoints());
        //System.out.println("PREDICTED: " + oCard.getName());

        // Agent is exhausted
        if(agent.getStaminaPoints() == 0) return new CardRest();

        // Other agent is defending
        if(oCard.getName().equals("cDefend")) return new CardRest();

        Card c = getStupidMove(stateBattle);
        //System.out.println(c.getName());

        // We have enough energy and health to attack
        return c;
    }

    private int calcDistanceBetweenAgents( StateBattle stateBattle ) {

        StateAgent asFirst = stateBattle.getAgentState( 0 );
        StateAgent asSecond = stateBattle.getAgentState( 1 );

        return Math.abs( asFirst.getCol() - asSecond.getCol() ) + Math.abs( asFirst.getRow() - asSecond.getRow() );
    }
}
