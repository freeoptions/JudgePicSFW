namespace JudgePicSFW.Models;

public enum LabelOrigin
{
    None = 0,
    SampleLibrary = 1,
    ModelPrediction = 2,
    ManualCorrection = 3,
    ConfirmedMove = 4,
    AiModel = 5,
    PersonalAi = 6,
}
