using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace persistenceLayer.Data.Migrations
{
    /// <inheritdoc />
    public partial class AddUserRelationshipToQuizAttempt : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_QuizAttempts_AspNetUsers_UserId",
                table: "QuizAttempts");

            migrationBuilder.DropIndex(
                name: "IX_QuizAttempts_UserId",
                table: "QuizAttempts");

            migrationBuilder.DropColumn(
                name: "UserId",
                table: "QuizAttempts");

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 533, DateTimeKind.Local).AddTicks(428));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 533, DateTimeKind.Local).AddTicks(504));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 533, DateTimeKind.Local).AddTicks(507));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 533, DateTimeKind.Local).AddTicks(509));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 533, DateTimeKind.Local).AddTicks(511));

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAENTbGgjdfrpKZZ15A/jsQ2pPI+sT1KOnJyDVZ1R3rUBks7+4xznDOYXf/T1H0YtcBw==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEGC9tTyCVwR7QIlFbBgf4aWR9ww3yH7vXzchQu0oeCGn9KVvmhTqilj3yMLZQBU+OQ==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 536, DateTimeKind.Local).AddTicks(7370));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 536, DateTimeKind.Local).AddTicks(7424));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 536, DateTimeKind.Local).AddTicks(7432));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 536, DateTimeKind.Local).AddTicks(7428));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 536, DateTimeKind.Local).AddTicks(7430));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2589));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2632));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2636));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2640));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2642));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 6,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2645));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 7,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2647));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 8,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2649));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 9,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2650));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 10,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2653));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 11,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2655));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 12,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2656));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 13,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2658));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 14,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2660));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 15,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2662));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 16,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2668));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 17,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2692));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 18,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2695));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 19,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2697));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 20,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2699));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 21,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2701));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 22,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2702));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 23,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2704));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 24,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2705));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 25,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2707));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 26,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2709));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 27,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2710));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 28,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2712));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 29,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2714));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 30,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2716));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 31,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2721));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 32,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2723));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 33,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2724));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 34,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2727));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 35,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2729));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 36,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2730));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 37,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2732));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 38,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2733));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 39,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2735));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 40,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2736));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 41,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2738));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 42,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2739));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 43,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2741));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 44,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2743));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 45,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2744));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 46,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2749));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 47,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2750));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 48,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2751));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 49,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2753));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 50,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2755));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 51,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2756));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 52,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2758));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 53,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2759));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 54,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2761));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 55,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2763));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 56,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2764));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 57,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2766));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 58,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2767));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 59,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2769));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 60,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2770));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 61,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2775));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 62,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2776));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 63,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2778));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 64,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2779));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 65,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2781));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 66,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2851));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 67,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2853));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 68,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2854));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 69,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2856));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 70,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2858));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 71,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2860));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 72,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2861));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 73,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2863));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 74,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2865));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 75,
                column: "CreatedAt",
                value: new DateTime(2026, 6, 19, 21, 18, 25, 537, DateTimeKind.Local).AddTicks(2866));

            migrationBuilder.AddForeignKey(
                name: "FK_QuizAttempts_AspNetUsers_StudentId",
                table: "QuizAttempts",
                column: "StudentId",
                principalTable: "AspNetUsers",
                principalColumn: "Id",
                onDelete: ReferentialAction.Restrict);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropForeignKey(
                name: "FK_QuizAttempts_AspNetUsers_StudentId",
                table: "QuizAttempts");

            migrationBuilder.AddColumn<string>(
                name: "UserId",
                table: "QuizAttempts",
                type: "nvarchar(450)",
                nullable: true);

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 553, DateTimeKind.Local).AddTicks(5248));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 553, DateTimeKind.Local).AddTicks(5312));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 553, DateTimeKind.Local).AddTicks(5316));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 553, DateTimeKind.Local).AddTicks(5320));

            migrationBuilder.UpdateData(
                table: "AcademicYear",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 553, DateTimeKind.Local).AddTicks(5322));

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "585c8473-10ce-4377-8407-1f64655876c1",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEHZw/IgtxNLGHzjQoIRw+XV7a113Th1g85amLE33S/iost4dCGhrYlllfHX85fo1VQ==");

            migrationBuilder.UpdateData(
                table: "AspNetUsers",
                keyColumn: "Id",
                keyValue: "6dc6528a-b280-4770-9eae-82671ee81ef7",
                column: "PasswordHash",
                value: "AQAAAAIAAYagAAAAEExqDcNGp8A/5nXduBOmcxwhsPoLDhdFJK0X8Ee00lWzYaDri9jnm90QD9252+aI8Q==");

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1000,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 557, DateTimeKind.Local).AddTicks(7697));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1001,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 557, DateTimeKind.Local).AddTicks(7733));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1002,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 557, DateTimeKind.Local).AddTicks(7828));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1003,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 557, DateTimeKind.Local).AddTicks(7737));

            migrationBuilder.UpdateData(
                table: "Departments",
                keyColumn: "Id",
                keyValue: 1004,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 557, DateTimeKind.Local).AddTicks(7825));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 1,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3743));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 2,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3781));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 3,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3786));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 4,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3790));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 5,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3793));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 6,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3798));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 7,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3800));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 8,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3802));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 9,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3805));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 10,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3808));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 11,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3810));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 12,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3813));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 13,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3815));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 14,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3817));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 15,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3819));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 16,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3826));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 17,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3828));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 18,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3831));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 19,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3833));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 20,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3835));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 21,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3837));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 22,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3839));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 23,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3841));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 24,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3843));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 25,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3845));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 26,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3847));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 27,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3848));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 28,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3850));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 29,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3852));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 30,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3854));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 31,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3860));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 32,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3862));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 33,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3863));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 34,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3866));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 35,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3868));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 36,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3870));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 37,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3872));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 38,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3955));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 39,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3958));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 40,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3960));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 41,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3962));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 42,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3965));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 43,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3968));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 44,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3970));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 45,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3972));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 46,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3979));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 47,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3981));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 48,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3983));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 49,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3986));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 50,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3988));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 51,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3990));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 52,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3992));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 53,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3994));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 54,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3996));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 55,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(3998));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 56,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4000));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 57,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4001));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 58,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4003));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 59,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4005));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 60,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4007));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 61,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4013));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 62,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4015));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 63,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4016));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 64,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4019));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 65,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4020));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 66,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4024));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 67,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4026));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 68,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4028));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 69,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4029));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 70,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4032));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 71,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4033));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 72,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4035));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 73,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4037));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 74,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4039));

            migrationBuilder.UpdateData(
                table: "Fee",
                keyColumn: "Id",
                keyValue: 75,
                column: "CreatedAt",
                value: new DateTime(2026, 5, 18, 17, 43, 27, 558, DateTimeKind.Local).AddTicks(4041));

            migrationBuilder.CreateIndex(
                name: "IX_QuizAttempts_UserId",
                table: "QuizAttempts",
                column: "UserId");

            migrationBuilder.AddForeignKey(
                name: "FK_QuizAttempts_AspNetUsers_UserId",
                table: "QuizAttempts",
                column: "UserId",
                principalTable: "AspNetUsers",
                principalColumn: "Id");
        }
    }
}
