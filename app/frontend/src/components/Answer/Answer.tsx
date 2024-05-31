import { useMemo, useState, useEffect } from "react";
import { Stack, IconButton } from "@fluentui/react";
import DOMPurify from "dompurify";

import styles from "./Answer.module.css";

import { ChatAppResponse, getCitationFilePath } from "../../api";
import { parseAnswerToHtml } from "./AnswerParser";
import { AnswerIcon } from "./AnswerIcon";
import { logFeedbackApi } from "../../api"; // Import the new function

interface Props {
    answer: ChatAppResponse;
    isSelected?: boolean;
    isStreaming: boolean;
    onCitationClicked: (filePath: string) => void;
    onThoughtProcessClicked: () => void;
    onSupportingContentClicked: () => void;
    onFollowupQuestionClicked?: (question: string) => void;
    showFollowupQuestions?: boolean;
    idToken?: string; // Add idToken to props if not already present
}

const extractPhrase = (data: ChatAppResponse) => {
    try {
        const description = data.choices[0].context.thoughts[0].description;
        const contentString = JSON.parse(description[description.length-1].replace(/'/g, '"'));
        const match = contentString.content.match(/Generate search query for: (.*)/);
        return match ? match[1] : null;
    } catch (e) {
        console.error("Error accessing nested data:", e);
        return null;
    }
};

export const Answer = ({
    answer,
    isSelected,
    isStreaming,
    onCitationClicked,
    onThoughtProcessClicked,
    onSupportingContentClicked,
    onFollowupQuestionClicked,
    showFollowupQuestions,
    idToken // Destructure idToken from props
}: Props) => {
    const followupQuestions = answer.choices[0].context.followup_questions;
    const messageContent = answer.choices[0].message.content;
    const question = extractPhrase(answer);
    const parsedAnswer = useMemo(() => parseAnswerToHtml(messageContent, isStreaming, onCitationClicked), [answer]);

    const sanitizedAnswerHtml = DOMPurify.sanitize(parsedAnswer.answerHtml);

    // State to manage feedback
    const [feedback, setFeedback] = useState<{ thumbsUp: boolean; thumbsDown: boolean }>({
        thumbsUp: false,
        thumbsDown: false,
    });

    // State to track if the answer has been logged
    const [answerLogged, setAnswerLogged] = useState<boolean>(false);

    // Log the answer whenever the streaming is complete or the answer changes
    useEffect(() => {
        const logInitialAnswer = async () => {
            if (!isStreaming && !answerLogged) {
                try {
                    await logFeedbackApi(messageContent, false, false, idToken, question);
                    setAnswerLogged(true);
                } catch (error) {
                    console.error('Error logging initial answer:', error);
                }
            }
        };

        logInitialAnswer();
    }, [messageContent, isStreaming, idToken, answerLogged, question]);

    const logFeedback = async (thumbsUp: boolean, thumbsDown: boolean) => {
        try {
            await logFeedbackApi(messageContent, thumbsUp, thumbsDown, idToken, question);
        } catch (error) {
            console.error('Error logging feedback:', error);
        }
    };

    const handleThumbsUpClick = () => {
        const newThumbsUp = !feedback.thumbsUp;
        setFeedback({ thumbsUp: newThumbsUp, thumbsDown: false });
        logFeedback(newThumbsUp, false);
    };

    const handleThumbsDownClick = () => {
        const newThumbsDown = !feedback.thumbsDown;
        setFeedback({ thumbsUp: false, thumbsDown: newThumbsDown });
        logFeedback(false, newThumbsDown);
    };

    return (
        <Stack className={`${styles.answerContainer} ${isSelected && styles.selected}`} verticalAlign="space-between">
            <Stack.Item>
                <Stack horizontal horizontalAlign="space-between">
                    <AnswerIcon />
                    <div>
                        <IconButton
                            style={{ color: "black" }}
                            iconProps={{ iconName: "Lightbulb" }}
                            title="Show thought process"
                            ariaLabel="Show thought process"
                            onClick={() => onThoughtProcessClicked()}
                            disabled={!answer.choices[0].context.thoughts?.length}
                        />
                        <IconButton
                            style={{ color: "black" }}
                            iconProps={{ iconName: "ClipboardList" }}
                            title="Show supporting content"
                            ariaLabel="Show supporting content"
                            onClick={() => onSupportingContentClicked()}
                            disabled={!answer.choices[0].context.data_points}
                        />
                        <IconButton
                            // Change color based on thumbsUp state
                            style={{ color: feedback.thumbsUp ? "green" : "black" }}
                            iconProps={{ iconName: "Like" }}
                            title="Thumbs Up"
                            ariaLabel="Thumbs Up"
                            onClick={handleThumbsUpClick}
                        />
                        <IconButton
                            // Change color based on thumbsDown state
                            style={{ color: feedback.thumbsDown ? "red" : "black" }}
                            iconProps={{ iconName: "Dislike" }}
                            title="Thumbs Down"
                            ariaLabel="Thumbs Down"
                            onClick={handleThumbsDownClick}
                        />
                    </div>
                </Stack>
            </Stack.Item>

            <Stack.Item grow>
                <div className={styles.answerText} dangerouslySetInnerHTML={{ __html: sanitizedAnswerHtml }}></div>
            </Stack.Item>

            {!!parsedAnswer.citations.length && (
                <Stack.Item>
                    <Stack horizontal wrap tokens={{ childrenGap: 5 }}>
                        <span className={styles.citationLearnMore}>Citations:</span>
                        {parsedAnswer.citations.map((x, i) => {
                            const path = getCitationFilePath(x);
                            return (
                                <a key={i} className={styles.citation} title={x} onClick={() => onCitationClicked(path)}>
                                    {`${++i}. ${x}`}
                                </a>
                            );
                        })}
                    </Stack>
                </Stack.Item>
            )}

            {!!followupQuestions?.length && showFollowupQuestions && onFollowupQuestionClicked && (
                <Stack.Item>
                    <Stack horizontal wrap className={`${!!parsedAnswer.citations.length ? styles.followupQuestionsList : ""}`} tokens={{ childrenGap: 6 }}>
                        <span className={styles.followupQuestionLearnMore}>Follow-up questions:</span>
                        {followupQuestions.map((x, i) => {
                            return (
                                <a key={i} className={styles.followupQuestion} title={x} onClick={() => onFollowupQuestionClicked(x)}>
                                    {`${x}`}
                                </a>
                            );
                        })}
                    </Stack>
                </Stack.Item>
            )}
        </Stack>
    );
};
