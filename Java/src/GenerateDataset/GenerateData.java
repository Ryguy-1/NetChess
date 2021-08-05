package GenerateDataset;

import java.io.*;
import java.util.ArrayList;
import java.util.Random;

public class GenerateData {

    Random random = new Random();

    // Plays out games randomly and saves every position bitboard only to an ArrayList
    // ArrayList structure = [[[Board, board,...], [2=cap, 0=lc, 1=stalemate]], ...]
    ArrayList<ArrayList<ArrayList<Long[]>>> results = new ArrayList<>();
    //         games    boards for game/results
    private int numGames = 0;

    private final int stalemateNumMoves = 50;

    //private final static String serialLocation = "Results\\results.ser";
    private final static String serialSaveVariable = "Results50Moves\\results"; //add #.ser for save number (batch saves) // was "Results\\results"
    private final int numPerBatch = 100; //was 500

    GenerateData(int numGames){
        this.numGames = numGames;
        generate();
    }

    private void generate(){
        int gamesPlayed = 0;
        int batchNum = 0;
        while(gamesPlayed!=numGames){
            //Get New Board
            Position currentPos = Runner.mainBoard.mainPosition.getPositionCopy();

            int movesMade = 0;

            //Store Boards For This Game
            ArrayList<Long[]> boards = new ArrayList<>();
            ArrayList<Long[]> result = new ArrayList<>();

            char turn = 'c';
            GAME_RUNNING:
            while (true) {

                //check if either is in checkmate or is stalemate
                if (Runner.search.capitalIsInCheckmate(currentPos)) {
                    //save result to ArrayList -> Lc Won = 0
                    Long[] resultArr = {0l};
                    result.add(resultArr);
                    break GAME_RUNNING;
                } else if (Runner.search.lowerCaseIsInCheckmate(currentPos)) {
                    //save result to ArrayList -> Cap Won = 2
                    Long[] resultArr = {2l};
                    result.add(resultArr);
                    break GAME_RUNNING;
                } else if (Runner.search.lowerCaseIsInStalemate(currentPos) || Runner.search.capitalIsInStalemate(currentPos)) {
                    //save result to ArrayList -> Stalemate = 1
                    Long[] resultArr = {1l};
                    result.add(resultArr);
                    break GAME_RUNNING;
                } else if (movesMade > stalemateNumMoves) {
                    //save result to ArrayList -> Stalemate = 1
                    Long[] resultArr = {3l};
                    result.add(resultArr);
                    break GAME_RUNNING;
                }

                //switch turn
                switch (turn) {
                    case 'c':
                        turn = 'l';
                        break;
                    case 'l':
                        turn = 'c';
                        break;
                }

                //Make Moves
                currentPos = makeMove(currentPos, turn);
                movesMade++;
                //save to ArrayList
                boards.add(currentPos.getCurrentBoard());

                if (movesMade % 10000 == 0) {

                    System.out.println("Moves Made = " + movesMade);
                }

            }

            //switch turn
            switch (turn) {
                case 'c':
                    turn = 'l';
                    break;
                case 'l':
                    turn = 'c';
                    break;
            }

            if (result.get(0)[0] != 3) {
                ArrayList<ArrayList<Long[]>> game = new ArrayList<>();
                game.add(boards);
                game.add(result);


                //add the game with boards and result to the member variable
                results.add(game);
                gamesPlayed++;
                //System.out.println("Games Played = " + gamesPlayed);
                String winnerString = "";
                if (result.get(0)[0] == 0) {
                    winnerString = "Capital";
                } else if (result.get(0)[0] == 1) {
                    winnerString = "Lower Case";
                } else {
                    winnerString = "Stalemate";
                }

                //System.out.println("Final Board: ===================== Winner = " + winnerString + ", Turn = " + turn);
                //Runner.mainBoard.drawGameBoard(currentPos.getCurrentBoard());


                if (gamesPlayed % numPerBatch == 0 && gamesPlayed != 0) {
                    String saveLocation = serialSaveVariable + batchNum + ".ser";
                    saveResults(saveLocation);
                    System.out.println("=================== Saved Batch " + batchNum + " ============== " + gamesPlayed + "/" + numGames);
                    batchNum++;
//                loadResults(saveLocation);
                }
            }
        }
    }

    private Position makeMove(Position pos, char casing){
        //generate possible moves
        String[] possibleMoves = Runner.search.getPossibleMovesByCasing(pos, casing);
        //get random next move
        String nextMove = possibleMoves[random.nextInt(possibleMoves.length)];
        //make the move
        pos.fromToMove(nextMove);
        //return a new position object for new position
        return pos.getPositionCopy();
    }


    private void saveResults(String location){
        //write to file
        try{
            FileOutputStream writeData = new FileOutputStream(location);
            ObjectOutputStream writeStream = new ObjectOutputStream(writeData);

            writeStream.writeObject(results);
            writeStream.flush();
            writeStream.close();

        }catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Results Size Before Save: " + results.size());
        results.clear();
        System.out.println("Results Size After Save: " + results.size());
    }

    public ArrayList<ArrayList<ArrayList<Long[]>>> loadResults(String location){
        ArrayList<ArrayList<ArrayList<Long[]>>> loadedResults = new ArrayList<>();
        try{
            FileInputStream readData = new FileInputStream(location);
            ObjectInputStream readStream = new ObjectInputStream(readData);

            loadedResults = (ArrayList<ArrayList<ArrayList<Long[]>>>) readStream.readObject();
            readStream.close();
        }catch (Exception e) {
            e.printStackTrace();
        }

        System.out.println("Results Size After Load: " + loadedResults.size());

        return loadedResults;
    }



}
