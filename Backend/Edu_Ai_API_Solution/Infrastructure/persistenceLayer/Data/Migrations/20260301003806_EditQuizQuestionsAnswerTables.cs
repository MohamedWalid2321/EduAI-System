using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class EditQuizQuestionsAnswerTables : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "IX_QuizQuestions_QuizId",
                table: "QuizQuestions");

            migrationBuilder.DropIndex(
                name: "IX_QuestionChoices_QuizQuestionId",
                table: "QuestionChoices");

            migrationBuilder.AddColumn<bool>(
                name: "IsActive",
                table: "Quizzes",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.AddColumn<int>(
                name: "QuizCode",
                table: "Quizzes",
                type: "int",
                nullable: false,
                defaultValue: 0);

            migrationBuilder.AddColumn<bool>(
                name: "IsActive",
                table: "QuizQuestions",
                type: "bit",
                nullable: false,
                defaultValue: false);

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEI8lL+1b8QQssH/g8bPb1WzbXknIp14CxyCoL7OQK6946VU9c8uAxUgLsQ7LQ9sTYg==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEC2KYt4ecW0x5EMAgZAbJtRR4COk4OxUVwyfl/eiQya0x/Ajx41WYMKcamAyV/P27w==");

            migrationBuilder.CreateIndex(
                name: "IX_Quizzes_QuizCode",
                table: "Quizzes",
                column: "QuizCode",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_QuizQuestions_QuizId_QuestionText",
                table: "QuizQuestions",
                columns: new[] { "QuizId", "QuestionText" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "IX_QuestionChoices_QuizQuestionId_ChoiceText",
                table: "QuestionChoices",
                columns: new[] { "QuizQuestionId", "ChoiceText" },
                unique: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropIndex(
                name: "IX_Quizzes_QuizCode",
                table: "Quizzes");

            migrationBuilder.DropIndex(
                name: "IX_QuizQuestions_QuizId_QuestionText",
                table: "QuizQuestions");

            migrationBuilder.DropIndex(
                name: "IX_QuestionChoices_QuizQuestionId_ChoiceText",
                table: "QuestionChoices");

            migrationBuilder.DropColumn(
                name: "IsActive",
                table: "Quizzes");

            migrationBuilder.DropColumn(
                name: "QuizCode",
                table: "Quizzes");

            migrationBuilder.DropColumn(
                name: "IsActive",
                table: "QuizQuestions");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEOW7LUkBTomeYWgO3ngzfwf86of175TF2H9qWOzfw2HafEYEKNw6sKzsvcBNI8UF+w==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEE5AVwCBYWNI0qsQ9rFLnsEvzM3WDU/24AM1jdjzE+bWCJ3+LkhxaTpqWXchYT7iRw==");

            migrationBuilder.CreateIndex(
                name: "IX_QuizQuestions_QuizId",
                table: "QuizQuestions",
                column: "QuizId");

            migrationBuilder.CreateIndex(
                name: "IX_QuestionChoices_QuizQuestionId",
                table: "QuestionChoices",
                column: "QuizQuestionId");
        }
    }
}
