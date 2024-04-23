import { Example } from "./Example";

import styles from "./Example.module.css";

const DEFAULT_EXAMPLES: string[] = [
    "How does solid content affect the particle morphology in spray drying?",
    "What is the effect of solution viscosity on atomization efficiency?",
    "How does the type of excipient influence the drying performance and product characteristics?",
    "How do inlet and outlet air temperatures affect the drying rate and product moisture content?",
    "How does pump speed and air speed influence the properties of large porous particles?",
    "What is the role of atomization pressure in achieving desired particle sizes and densities?",
    "How does the drying temperature control the desulfurization efficiency in flue gas treatment?",
    "How can optimization techniques and mathematical models enhance the quality and efficiency of spray drying processes?",
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
