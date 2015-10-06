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
     * @param oCard             The predicted enemy card / null
     * @return attack card, null if none found
     */
    private Card attack( StateBattle stateBattle, Card oCard){
        Card[] move = new Card[2];
        if(learned) move[m_noOpponentAgent] = oCard;
        else move[m_noOpponentAgent] = new CardRest();

        Card bestCard = null;
        int bestSTAT = 0;

        ArrayList<Card> cards = m_deck.getCards(stateBattle.getAgentState( m_noThisAgent ).getStaminaPoints());
        // Find attack cards
        for ( Card card : cards ) {
            if ( card.getType() == Card.CardActionType.ctAttack) {

                StateBattle bs = (StateBattle) stateBattle.clone();   // clone the state, as play( ) modifies it.
                move[m_noThisAgent] = card;
                bs.play( move );
                StateAgent agent = bs.getAgentState( m_noThisAgent );
                StateAgent oAgent  = bs.getAgentState(m_noOpponentAgent);

                // In range?
                if(card.inAttackRange(agent.getCol(), agent.getRow(), oAgent.getCol(), oAgent.getRow())) {
                    if (oAgent.getHealthPoints() - card.getHitPoints() <= 0) return card; // Killing move?
                    // Find strongest attack card
                    if (card.getHitPoints() + card.getDefencePoints() > bestSTAT) {
                        bestSTAT = card.getHitPoints() + card.getDefencePoints();
                        bestCard = card;
                    }
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
     * @return a card moving the agent away from the opponent / null if none found
     */
    private Card moveAway(StateBattle stateBattle, Card oCard){
        Card [] move = new Card[2];
        if(learned) move[m_noOpponentAgent] = oCard;    // Predict where opponent moves
        else move[m_noOpponentAgent] = new CardRest();  // Assume opponent will remain still

        Card bestCard = null;
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
     * @return a card moving agent closer to opponent / null if none found
     */
    private Card moveCloser( StateBattle stateBattle, Card oCard){
        Card [] move = new Card[2];
        if(learned) move[m_noOpponentAgent] = oCard;    // Predict where opponent moves
        else move[m_noOpponentAgent] = new CardRest();  // Assume opponent will remain still

        Card bestCard = null;
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
     * Returns the card that gives the highest amount of defence points, with the maximum cost of "stamina"
     * @param stamina       Maximum cost
     * @return a defence card / null if none found
     */
    private Card defend(int stamina){
        Card bestCard = null;
        int bestDefence = 0;

        ArrayList<Card> cards = m_deck.getCards(stamina);
        // Find best defence card
        for (Card card : cards ){
            if(bestDefence < card.getDefencePoints()){
                bestDefence = card.getDefencePoints();
                bestCard = card;
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
            if((c = attack(stateBattle, null)) != null) return c;
            else if((c = moveCloser(stateBattle, null)) != null) return c;
        }
        else{
            Card c;
            if((c = moveAway(stateBattle, null)) != null) return c;
        }
        return new CardRest();
    }

    /**
     * Gets a move, based on what is learned by using the classifier.
     * Strategy:
     *      Stamina = 0 : Rest
     *      Enemy is defending: see if we have plenty of stamina, then if we
     *          can penetrate his defense
     *      Enemy is moving (or healing): If we have more HP we are likely in a better standing,
     *          so we use our heaviest attack or chase him if he moves out of range
     *      Enemy is attacking: We try to block the attack, if it wastes more of his stamina, and completely
     *          blocks the attack. Or we try to counterattack if we can at least match his power. Or we try to
     *          dodge his attack by moving away from it.
     *      We throw our heaviest punch, or rest (if out of stamina, or he is out of range)
     *
     * @param stateBattle       Current state of game
     * @param oCard             Predicted opponent move
     *
     * @return the move performed by the agent
     */

    private Card getLearnedMove(StateBattle stateBattle, Card oCard){
        StateAgent agent = stateBattle.getAgentState(m_noThisAgent);
        StateAgent oAgent = stateBattle.getAgentState(m_noOpponentAgent);
        //System.out.println("HP: " + agent.getHealthPoints() + ", EHP: " + oAgent.getHealthPoints());
        //System.out.println("PREDICTED: " + oCard.getName());

        // Agent is exhausted, it needs to rest
        if(agent.getStaminaPoints() == 0) return new CardRest();

        Card attCard = attack(stateBattle, oCard);
        Card defense = defend(oCard.getStaminaPoints());

        // Other agent is defending, rest up! (or attack if worth it)
        if(oCard.getType() == Card.CardActionType.ctDefend){
            if(agent.getStaminaPoints() + (new CardRest()).getStaminaPoints() > StateAgent.MAX_STAMINA)
                if(attCard.getHitPoints() > defense.getDefencePoints())
                    return attCard;     // Attack if plenty of stamina, and it penetrates the defence
            return new CardRest();      // Rest up
        }

        Card mvcCard = moveCloser(stateBattle, oCard);  // Move closer card
        Card mvfCard = moveAway(stateBattle, oCard); // Move away card

        // Other agent is moving (or resting)
        if(oCard.getType() == Card.CardActionType.ctMove){
            // Do we have the upper hand
            if(agent.getHealthPoints() >= oAgent.getHealthPoints()){
                // Then chase, go for the neck!
                if(attCard != null) return attCard;         // Attack if possible
                else if (mvcCard != null) return mvcCard;   // Chase if possible
            }
            else{
                // Then lets go for the tie
                if(mvfCard != null) return mvfCard;         // Run if possible
            }
        }
        // Other agent is attacking
        if(oCard.getType() == Card.CardActionType.ctAttack){
            // Can we make the attack a waste?
            if(defense != null &&
                    oCard.getHitPoints() - defense.getDefencePoints() >= 0 &&
                    oCard.getStaminaPoints() > defense.getStaminaPoints()) return defense;
            // Can we equal the damage dealt (or do more)?
            else if(attCard != null && oCard.getHitPoints() <= attCard.getHitPoints())return attCard;
            // Lets try and dodge the attack by moving away
            return mvfCard;
        }

        // Didn't find any match move according to the tactic, lets attack if possible, otherwise rest
        return (attCard != null) ? attCard : new CardRest();
    }

    private int calcDistanceBetweenAgents( StateBattle stateBattle ) {

        StateAgent asFirst = stateBattle.getAgentState( 0 );
        StateAgent asSecond = stateBattle.getAgentState( 1 );

        return Math.abs( asFirst.getCol() - asSecond.getCol() ) + Math.abs( asFirst.getRow() - asSecond.getRow() );
    }
}
