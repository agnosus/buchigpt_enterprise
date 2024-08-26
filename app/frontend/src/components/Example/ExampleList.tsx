import { Example } from "./Example";

import styles from "./Example.module.css";

const DEFAULT_EXAMPLES: string[] = [
    "What particle sizes are used in FlashPure cartridges?",
    "What are the dimensions for columns in the SFC-250 ?",
    "What is the maximum operating pressure for the  SFC-660 ?",
    "What is the purpose of the add-on pump in SFC instruments?",
    "What phase types are used in PrepPure columns?",
    "What is the function of the FlashPure cartridge holder?",
    "What sample injection options are available for the Pure?",
    "What is the purpose of the mixing chambers in the Pure?",
    "What is the function of the solid loader in the Pure?",
    "What is the maximum sample size for the solid loader in the Pure?",
    "What are the key features of the Pure C-830 Prep chromatography?",
    "What are the selectable DAD wavelengths in Sepiatec SFC instruments?",
    "What is the function of the chiller in Sepiatec SFC instruments?",
    "What type of phases are used in  SFC columns?"
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
