import { Example } from "./Example";

import styles from "./Example.module.css";

const DEFAULT_EXAMPLES: string[] = [
    "What are the dimensions of the R-80?",
    "My R-300 cant reach the desired vacuum. What should I do?",
    "how does the foam sensor work?",
    "That is the largest flask size that can be used with the Rotavapor R-100?",
    "Can Silicon oil be used as a heating bath fluid on the Rotavapor R-300?"    
];

const GPT4V_EXAMPLES: string[] = [
    "Compare the impact of interest rates and GDP in financial markets.",
    "What is the expected trend for the S&P 500 index over the next five years? Compare it to the past S&P 500 performance",
    "Can you identify any correlation between oil prices and stock market trends?"
];

interface Props {
    onExampleClicked: (value: string) => void;
    useGPT4V?: boolean;
}

export const ExampleList = ({ onExampleClicked}: Props) => {
    const randomExamples = DEFAULT_EXAMPLES.sort(() => 0.5 - Math.random()).slice(0, 3);
    return (
        <ul className={styles.examplesNavList}>
            {randomExamples.map((question, i) => (
                <li key={i}>
                    <Example text={question} value={question} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
