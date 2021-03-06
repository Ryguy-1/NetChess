package GenerateDataset;

import com.sun.scenario.effect.impl.sw.sse.SSEBlend_ADDPeer;
import sun.management.GarbageCollectionNotifInfoCompositeData;

import java.util.ArrayList;
import java.util.Arrays;

public class Runner {
    public static MainBoard mainBoard;
    public static CheckValidConditions checkValidConditions;
    public static BitboardControlAndSeparation controlAndSeparation;
    public static Search search;
    public static GenerateData generateData;

    public static void main(String[] args){
        //initialize mainBoard FIRST
        mainBoard = new MainBoard();
        checkValidConditions = new CheckValidConditions();
        controlAndSeparation = new BitboardControlAndSeparation();
        search = new Search();
        generateData = new GenerateData(1000000);
//        ArrayList<ArrayList<ArrayList<Long[]>>> loaded_arr = generateData.loadResults("ResultsZeroCentered\\results0.ser");
//        for (int i = 0; i < loaded_arr.size(); i++) { // Each Game
//            System.out.println("======================== Game " + i + " ===================================");
//            for (int j = 0; j < loaded_arr.get(i).size(); j++) { // Each Board arraylist followed by arraylist with the result
//                for (int k = 0; k < loaded_arr.get(i).get(0).size(); k++) { // Print out each game board
//                    System.out.println("Move " + k);
//                    System.out.println(loaded_arr.get(i).get(0).size());
//                    Long[] thisBoard = loaded_arr.get(i).get(0).get(k);
//                    mainBoard.drawGameBoard(thisBoard); // Draw Each Game Board
//                }
//                System.out.println("Result = " + loaded_arr.get(i).get(1).get(0)[0]);
//
//            }
//        }




//        mainBoard.drawGameBoard(mainBoard.mainPosition.getCurrentBoard());




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//NEXT: Beat 1500: Handily with 77 percent accuracy vs its 25 percent.

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//        mainBoard.mainPosition.setCapitalAFileRookHasMoved(true);
//        mainBoard.mainPosition.setCapitalHFileRookHasMoved(true);
//        mainBoard.mainPosition.setLowerCaseAFileRookHasMoved(true);
//        mainBoard.mainPosition.setLowerCaseHFileRookHasMoved(true);
//        mainBoard.mainPosition.setCapitalKingHasMoved(true);
//        mainBoard.mainPosition.setLowerCaseKingHasMoved(true);
//        mainBoard.drawGameBoard(mainBoard.mainPosition.getCurrentBoard());
//        //to make a move, do it here for testing...
//        long thisPiece = mainBoard.parseLong("0000000000000000000000000000000000000000000000000000000000000000", 3); //if you want to reference a specific piece, just change a bit in this long and use the reference

        //System.out.println(controlAndSeparation.splitBitboard(checkValidConditions.getPseudoLegalMoves(mainBoard.mainPosition, 'c')).length);


//        System.out.println();
//        //call minimax
//        Position bestPositionEvaluated = minimax.minimax(mainBoard.mainPosition, 1, true, Minimax.MIN, Minimax.MAX);
//
//        //print out some values from the minimax evaluation
//        System.out.println(bestPositionEvaluated.getMovesToCurrent());
//        System.out.println("Best Move: " + bestPositionEvaluated.getMovesToCurrent().get(0));
//        System.out.println();
//        for (int i = 0; i < bestPositionEvaluated.getMovesToCurrent().size(); i++) {
//            mainBoard.mainPosition.fromToMove(bestPositionEvaluated.getMovesToCurrent().get(i));
//            mainBoard.drawGameBoard(mainBoard.mainPosition.getCurrentBoard());
//        }
//
//        //////////////////////////////////////////
//        mainBoard.drawGameBoard(mainBoard.mainPosition.getCurrentBoard());
//        System.out.println("Ranking: " + boardEvaluation.getBoardRanking(mainBoard.mainPosition));
//        System.out.println("Capital is in Checkmate: " + search.capitalIsInCheckmate(mainBoard.mainPosition));
//        System.out.println("Lower Case is in Checkmate: " + search.lowerCaseIsInCheckmate(mainBoard.mainPosition));
//        System.out.println();
//        System.out.println("Capital is in Check: " + search.capitalIsInCheck(mainBoard.mainPosition));
//        System.out.println("Lower Case is in Check: " + search.lowerCaseIsInCheck(mainBoard.mainPosition));
//        System.out.println();
//        System.out.println("Capital is in Stalemate: " + search.capitalIsInStalemate(mainBoard.mainPosition));
//        System.out.println("Lower Case is in Stalemate: " + search.lowerCaseIsInStalemate(mainBoard.mainPosition));



//        mainBoard.visualizeBitboardArray(mainBoard.mainPosition.getCurrentBoardHistory());
//        mainBoard.visualizeBitboardArray(mainBoard.mainPosition.getCurrentBoard());

    }
}